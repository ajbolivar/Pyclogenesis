import os
import math
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
import datetime
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

# Subset storms based on geographic region
def subset_storms(region, storms):
    gdf = gpd.GeoDataFrame(storms, geometry=gpd.points_from_xy(storms.gen_lon, storms.gen_lat))

    # Define ocean basin polygons
    NA = Polygon([(-83,10),(-93,17),(-98,17),(-98,45),(-15,45),(-15,10)])
    
    try:
        # Global
        if region == "GL": pass
        # North Atlantic
        elif region == "NA": storms = gdf[gdf.geometry.within(NA)].drop(columns="geometry")
        # East Pacific
        elif region == "EP": pass
        # West Pacific
        elif region == "WP": pass
        # Northern Indian Ocean
        elif region == "NIO": pass
        # Southern Indian Ocean
        elif region == "SIO": pass
        # Oceania
        elif region == "OC": pass

    except:
        raise ValueError("Invalid region specified. Please select a region from the following options: 'GL', 'NA', 'EP', 'WP', 'NIO', 'SIO', 'OC'")
    
    return storms
    
def count_storms(storms, landfalls):
    storm_count = 0
    landfall_count = 0
    
    for enum in np.unique(storms.index.get_level_values('enum')):
        ens_storms = storms.xs(enum, level='enum', drop_level=False)
        ens_landfalls = landfalls.xs(enum, level='enum', drop_level=False)
        
        storm_count += len(np.unique(ens_storms.index.get_level_values('sid')))
        landfall_count += len(np.unique(ens_landfalls.index.get_level_values('sid')))
        
    nonlandfall_count = storm_count - landfall_count
    
    return storm_count, landfall_count, nonlandfall_count


def calc_pace(storms, num_years, nan_value, scale, ace = False, wind_units = 'kts'):
    PACE = []

    # Coefficients for PACE calculation from Zarzycki et al. 2021
    a1, a2, a3 = -5.46649378e-05, 1.15460199e+00, 3.38263470e+01
    # a1, a2, a3 = -2.81220736e-05,  5.93978581e-01,  1.74017763e+01
    
    # Find timestep in data to determine how many rows to skip for calculation
    try:
        t1 = storms.index.get_level_values('time')[1]
        t2 = storms.index.get_level_values('time')[2]
        dt = (t2 - t1).seconds // 3600
    except: 
        t1 = storms.HOUR[1] 
        t2 = storms.HOUR[2]
        dt = (t2 - t1)

    skip = int(6 / dt)

    # Subset storm dataset by ensemble member
    num_members = 0
    for enum in np.unique(storms.index.get_level_values('enum')):
        ens_df = storms.xs(enum, level='enum', drop_level=False)
        num_members += 1
        for sid in np.unique(ens_df.index.get_level_values('sid')):
            storm_df = ens_df.xs(sid, level='sid')
            storm_df = storm_df[::skip]
            
            # If there is missing pressure information, calculate ACE instead
            if ((storm_df.pressure.values == nan_value).any() == True) or (ace):
                # If there is missing wind information, skip all calculations
                if ((storm_df.wind.values == nan_value).any() == True):
                    continue
                conversion_factor = 1
                if wind_units == 'ms':
                    conversion_factor = 1.944
                elif wind_units == 'mph':
                    conversion_factor = 1 / 1.151
                
                storm_wind_kts = storm_df.wind.values * conversion_factor
                storm_ACE = (0.0001) * np.nansum((storm_wind_kts)**2)
                PACE.append(storm_ACE)
                continue

            # Terms for PACE equation
            delta_p = 1010-storm_df.pressure.values
            a1_term = a1*(delta_p**2)
            a2_term = a2*(delta_p)
            
            # Calculate storm PACE
            storm_PACE = (0.0001) * np.nansum((a1_term + a2_term + a3)**2)
            PACE.append(storm_PACE)
            
    # Sum ACE for all storms, average by number of years, then by number of ensemble members
    avg_annual_PACE = np.round(((np.nansum(PACE) / num_years) * scale) / num_members, 1)

    return avg_annual_PACE
    

def calc_lifetime_lmi_minpres(storms, num_years, nan_value, subset):
    LMI = []
    SLT = []
    min_pres = []

    num_members = 0
    for enum in np.unique(storms.index.get_level_values('enum')):
        ens_df = storms.xs(enum,level='enum',drop_level=False)
        num_members += 1
        for sid in np.unique(ens_df.index.get_level_values('sid')):
            storm_df = ens_df.xs(sid,level='sid')
            if np.min(storm_df.pressure) != nan_value:
                lmi = np.abs((storm_df[storm_df.pressure==np.min(storm_df.pressure)]).lat.values[0])
                min_pres.append(np.min(storm_df.pressure))
                LMI.append(lmi)
            
            # Record storm lifetime in days
            try:
                dates = len(pd.to_datetime(storm_df.index.get_level_values('time')).map(pd.Timestamp.date).unique())
            except:
                dates = math.ceil(len(storm_df.HOUR.values) / 24)
                
            SLT.append(dates)

    avg_LMI      = round(np.mean(LMI), 1)
    avg_SLT      = round(np.mean(SLT), 1)
    med_SLT      = round(np.median(SLT), 1)
    annual_TCD   = round((np.sum(SLT) / num_years / num_members), 1)
    avg_min_pres = round(np.mean(min_pres), 1)

    if subset: return avg_LMI, avg_min_pres
    else: return avg_LMI, avg_SLT, med_SLT, annual_TCD, avg_min_pres


def calc_lfpres(landfalls,num_years,nan_value):
    lf_pres = []
    
    for enum in np.unique(landfalls.index.get_level_values('enum')):
        ens_df = landfalls.xs(enum, level='enum', drop_level=False)
        for i in range(len(ens_df)):
            landfall_df = ens_df.iloc[i]
            if landfall_df.pressure > 850:
                lf_pres.append(landfall_df.pressure)
                
    avg_lf_pres = round(np.mean(lf_pres), 1)
    
    return avg_lf_pres
            
# Creates and outputs a csv containing several climatological TC statistics.
def storm_statistics(storms, landfalls, name, region='GL', start_year=1979, end_year=2014, 
                     nan_value=0, sdd=False, obs=False, subset=False, scale=1, sdd_timestep='6-hourly'):
    # Number of ensemble members
    num_members = len(np.unique(storms.index.get_level_values('enum')))
    # Number of years based on start_year and end_year (user defined in case dataset contains no storms in the bounding years)
    num_years = len(np.arange(start_year, end_year + 1, 1))
    year_range = np.arange(start_year, end_year + 1, 1)
    if (obs) or (not sdd):
        scale = 1 # scale should always be 1 if not using SDD model output
        storms = storms[storms.index.get_level_values('sid').str[0:4].astype('int').isin(year_range)]
        landfalls = landfalls[landfalls.index.get_level_values('sid').str[0:4].astype('int').isin(year_range)]
        if obs: wind_units = 'kts'
        else: wind_units = 'ms'
        
    # Subset storms by time
    elif sdd:
        wind_units = 'ms'
        try:
            storms = storms[storms.index.get_level_values('time').year.isin(year_range)]
            landfalls = landfalls[landfalls.index.get_level_values('time').year.isin(year_range)]
        except:
            storms = storms[storms.YEAR.isin(year_range)]
            landfalls = landfalls[landfalls.YEAR.isin(year_range)]

    # Get storm counts
    storm_count, landfall_count, nonlandfall_count = count_storms(storms, landfalls)

    # Annual average storm counts
    annual_storm_count = round(storm_count / num_years / num_members, 1)
    annual_landfall_count = round(landfall_count / num_years / num_members, 1)
    annual_nonlandfall_count = round(annual_storm_count - annual_landfall_count, 1)
    
    # Calculate percentage of landfalling/nonlandfalling TCs
    landfall_percent = round((landfall_count / storm_count) * 100, 1)
    nonlandfall_percent = round(100 - landfall_percent, 1)
    
    # PACE, LMI, SLT, TCD, and pressure
    ace = False
    annual_PACE = calc_pace(storms, num_years, nan_value, scale, ace, wind_units)
    avg_lf_pres = calc_lfpres(landfalls, num_years, nan_value)
    if subset:
        ace = True
        annual_ACE = calc_pace(storms, num_years, nan_value, scale, ace, wind_units)
        avg_LMI, avg_min_pres = calc_lifetime_lmi_minpres(storms, num_years, nan_value, subset)
        stats = [name, avg_LMI, annual_PACE, annual_ACE, landfall_percent, nonlandfall_percent, avg_min_pres, avg_lf_pres]
        stats = pd.DataFrame([stats],columns=['name', 'avg_LMI',  'PACE_yr', 'ACE_yr', 'LF_percent', 'NLF_percent', 'avg_MIN_pressure', 'avg_LF_pressure',])
    else: 
        avg_LMI, avg_SLT, med_SLT, tc_days, avg_min_pres = calc_lifetime_lmi_minpres(storms, num_years, nan_value, subset)
        stats = [name, annual_storm_count, tc_days, annual_PACE, avg_SLT, med_SLT, avg_LMI,
        annual_landfall_count,annual_nonlandfall_count,landfall_percent,nonlandfall_percent, avg_min_pres, avg_lf_pres]
        stats = pd.DataFrame([stats],columns=['name', 'storm_count_yr', 'TC_days_yr', 'PACE_yr', 'avg_life', 'med_life',
                                              'avg_LMI', 'LF_count_yr', 'NLF_count_yr', 'LF_percent', 'NLF_percent',
                                              'avg_MIN_pressure', 'avg_LF_pressure',])
    
    os.makedirs('tc_stats', exist_ok=True)
    # Save stats to csv
    stats.to_csv('tc_stats/{}_TC_stats_{}.csv'.format(name, region),index=False)
    
    return stats

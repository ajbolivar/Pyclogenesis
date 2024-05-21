import os
import math
import numpy as np
import xarray as xr
import pandas as pd
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

def region_bounds(region, longitudes, latitudes):
    ''' Determines whether storm genesis points are within a specified geographic region.
        Parameters: 
            region (str): oceanic domain that defines region bounds (options = 'GL','NA','EP','WP','IO','OC').
            longitudes (int arr): longitude(s) of storm genesis point(s) to be checked.
            lattitudes (int arr): latitude(s) of storm genesis point(s) to be checked.
            
        Returns:
            in_region (bool arr): True if point is in region, False if point is not in region.
    '''
    # Convert single lon/lat values to lists so that list comprehension code doesn't break
    if isinstance(longitudes,np.float64) and isinstance(latitudes,np.float64):
        longitudes = [longitudes]
        latitudes  = [latitudes]
        
    # Define ocean basin polygons
    NA = Polygon([(-70,0),(-92,17),(-98,17),(-98,90),(5,90),(5,0)])
    
    try:
        # Global
        if region == "GL":
            return[True for lon, lat in zip(longitudes, latitudes)]

        # North Atlantic
        if region == "NA":
            return [NA.contains(Point(lon, lat)) for lon,lat in zip(longitudes, latitudes)]

        # East Pacific
        if region == "EP":
            return [(True if -180 <= lon <= -70 and 0 <= lat <= 60 else False) for lon, lat in zip(longitudes, latitudes)]

        # West Pacific
        if region == "WP":
            return [(True if 100 <= lon <= 180 and 0 <= lat <= 60 else False) for lon, lat in zip(longitudes, latitudes)]

        # Northern Indian Ocean
        if region == "NIO":
            return [(True if 30 <= lon <= 100 and 0 <= lat <= 50 else False) for lon, lat in zip(longitudes, latitudes)]

        # Southern Indian Ocean
        if region == "SIO":
            return [(True if -20 <= lon <= 100 and -45 <= lat <= 0 else False) for lon, lat in zip(longitudes, latitudes)]

        # Oceania
        if region == "OC":
            return [(True if 100 <= lon <= 180 and -45 <= lat <= 0 else False) for lon, lat in zip(longitudes, latitudes)]

    except:
        raise ValueError("Invalid region specified. Please select a region from the following options: 'GL', 'NA', 'EP', 'WP', 'NIO', 'SIO', 'OC'")
    
    
def subset_storms(region, storms):
    in_region = region_bounds(region, storms.GEN_LON.values, storms.GEN_LAT.values)
    
    return storms.loc[in_region]
    
def count_storms(storms, landfalls):
    storm_count = 0
    landfall_count = 0
    
    for enum in np.unique(storms.index.get_level_values('ENUM')):
        ens_storms = storms.xs(enum, level='ENUM', drop_level=False)
        ens_landfalls = landfalls.xs(enum, level='ENUM', drop_level=False)
        
        storm_count += len(np.unique(ens_storms.index.get_level_values('SID')))
        landfall_count += len(np.unique(ens_landfalls.index.get_level_values('SID')))
        
    nonlandfall_count = storm_count - landfall_count
    
    return storm_count, landfall_count, nonlandfall_count


def calc_pace(storms, num_years, nan_value):
    PACE = []

    # Coefficients for PACE calculation from Zarzycki et al. 2021
    a1, a2, a3 = -5.46649378e-05, 1.15460199e+00, 3.38263470e+01
    
    # Find timestep in data to determine how many rows to skip for calculation
    dt   = storms.index.get_level_values('ISO_TIME')[3].hour - storms.index.get_level_values('ISO_TIME')[2].hour
    skip = int(6 / dt)
    
    # Subset storm dataset by ensemble member
    num_members = 0
    for enum in np.unique(storms.index.get_level_values('ENUM')):
        ens_df = storms.xs(enum, level='ENUM', drop_level=False)
        num_members += 1
        for sid in np.unique(ens_df.index.get_level_values('SID')):
            storm_df = ens_df.xs(sid, level='SID')
            storm_df = storm_df[::skip]
            
            # If there is missing pressure information, calculate ACE instead
            if (storm_df.PRES.values == nan_value).any() == True:
                storm_wind_kts = storm_df.WIND.values * 1.944
                storm_ACE = (0.0001) * np.nansum((storm_wind_kts)**2)
                PACE.append(storm_ACE)
                continue
                
            # Terms for PACE equation
            delta_p = 1010-storm_df.PRES.values
            a1_term = a1*(delta_p**2)
            a2_term = a2*(delta_p)
            
            # Calculate storm PACE
            storm_PACE = (10**-4) * np.nansum((a1_term + a2_term + a3)**2)
            PACE.append(storm_PACE)
    
    # Sum ACE for all storms, average by number of years, then by number of ensemble members
    avg_annual_PACE = round(((np.nansum(PACE)/num_years)/num_members), 1)
    
    return avg_annual_PACE

def calc_ace(storms, num_years, nan_value):
    ACE = []
    
    # Find timestep in data to determine how many rows to skip for calculation
    dt = storms.index.get_level_values('ISO_TIME')[3].hour - storms.index.get_level_values('ISO_TIME')[2].hour
    skip = int(6 / dt)
    
    # Subset storm dataset by ensemble member
    num_members = 0
    for enum in np.unique(storms.index.get_level_values('ENUM')):
        ens_df = storms.xs(enum,level='ENUM',drop_level=False)
        num_members += 1
        for sid in np.unique(ens_df.index.get_level_values('SID')):
            storm_df = ens_df.xs(sid,level='SID')
            storm_df = storm_df[::skip]
            
            # If no wind information available for storm, skip ACE calculation
            if (storm_df.WIND.values == nan_value).all() == True:
                continue
            
            # Convert from m/s to kts
            storm_wind_kts = storm_df.WIND.values * 1.944
            
            # Calculate storm ACE
            storm_ACE = (10**-4) * np.nansum((storm_wind_kts)**2)
            ACE.append(storm_ACE)
    
    # Sum ACE for all storms, average by number of years, then by number of ensemble members
    avg_annual_ACE = round(((np.nansum(ACE)/num_years)/num_members),1)
    
    return avg_annual_ACE
    

def calc_lifetime_lmi_minpres(storms, num_years, nan_value):
    LMI = []
    SLT = []
    min_pres = []

    num_members = 0
    for enum in np.unique(storms.index.get_level_values('ENUM')):
        ens_df = storms.xs(enum,level='ENUM',drop_level=False)
        num_members += 1
        for sid in np.unique(ens_df.index.get_level_values('SID')):
            storm_df = ens_df.xs(sid,level='SID')
            if np.min(storm_df.PRES) != nan_value:
                lmi = np.abs((storm_df[storm_df.PRES==np.min(storm_df.PRES)]).LAT.values[0])
                min_pres.append(np.min(storm_df.PRES))
                LMI.append(lmi)
            
            # Record storm lifetime in days
            try:
                dates = len(np.unique(storm_df.index.get_level_values('ISO_TIME').date))
            except:
                dates = math.ceil(len(storm_df.HOUR.values)/24)
                
            SLT.append(dates)

    avg_LMI      = round(np.mean(LMI),1)
    avg_SLT      = round(np.mean(SLT),1)
    med_SLT      = round(np.median(SLT),1)
    annual_TCD   = round((np.sum(SLT)/num_years/num_members),1)
    avg_min_pres = round(np.mean(min_pres),1)
    
    return avg_LMI, avg_SLT, med_SLT, annual_TCD, avg_min_pres


def calc_lfpres(landfalls,num_years,nan_value):
    lf_pres = []
    
    for enum in np.unique(landfalls.index.get_level_values('ENUM')):
        ens_df = landfalls.xs(enum,level='ENUM',drop_level=False)
        for i in range(len(ens_df)):
            landfall_df = ens_df.iloc[i]
            if landfall_df.PRES > 850:
                lf_pres.append(landfall_df.PRES)
                
    avg_lf_pres = round(np.mean(lf_pres),1)
    
    return avg_lf_pres
            
    
def storm_statistics(storms, landfalls, nonlandfalls, name, region='GL',
                     start_year=1979,end_year=2014,nan_value=0):
    ''' Creates a csv containing several climatological TC statistics.
        Parameters: 
        storms (pandas.DataFrame):
        landfalls (pandas.DataFrame:
        nonlandfalls (pandas.DataFrame)
        name (str):
        region (str):
        start_year (int)
        end_year (int):
        nan_value (int):
        
    Returns:
        None
    '''
    # Number of ensemble members
    num_members = len(np.unique(storms.index.get_level_values('ENUM')))
    # Number of years based on start_year and end_year (user defined in case dataset contains no storms in the bounding years)
    num_years = len(np.arange(start_year, end_year + 1, 1))
    
    # Subset storms by time
    storms = storms[storms.index.get_level_values('SID').str[0:4].astype('int').isin(np.arange(start_year, end_year + 1, 1))]
    landfalls = landfalls[landfalls.index.get_level_values('SID').str[0:4].astype('int').isin(np.arange(start_year, end_year + 1, 1))]
    nonlandfalls = nonlandfalls[nonlandfalls.index.get_level_values('SID').str[0:4].astype('int').isin(np.arange(start_year, end_year + 1, 1))]
    
    # Subset storms by basin
    storms       = subset_storms(region, storms)
    landfalls    = subset_storms(region, landfalls)
    nonlandfalls = subset_storms(region, nonlandfalls)
    
    # Get storm counts
    storm_count,landfall_count,nonlandfall_count = count_storms(storms, landfalls)

    # Annual average storm counts
    annual_storm_count = round(storm_count / num_years / num_members, 1)
    annual_landfall_count = round(landfall_count / num_years / num_members, 1)
    annual_nonlandfall_count = round(annual_storm_count - annual_landfall_count, 1)
    
    # Calculate percentage of landfalling/nonlandfalling TCs
    landfall_percent = round((landfall_count/storm_count)*100, 1)
    nonlandfall_percent = round(100 - landfall_percent, 1)
    
    # PACE, LMI, SLT, TCD, and pressure
    annual_PACE = calc_pace(storms, num_years, nan_value)
    avg_LMI, avg_SLT, med_SLT, tc_days, avg_min_pres = calc_lifetime_lmi_minpres(storms, num_years, nan_value)
    avg_lf_pres = calc_lfpres(landfalls, num_years, nan_value)

    stats = [name, annual_storm_count, tc_days, annual_PACE, avg_SLT, med_SLT, avg_LMI,
    annual_landfall_count,annual_nonlandfall_count,landfall_percent,nonlandfall_percent, avg_min_pres, avg_lf_pres]
    stats = pd.DataFrame([stats],columns=['NAME', 'STORM_COUNT_YR', 'TC_DAYS_YR', 'PACE_YR', 'AVG_LIFE', 'MED_LIFE',
                                          'AVG_LMI', 'LF_COUNT_YR', 'NLF_COUNT_YR', 'LF_PERCENT', 'NLF_PERCENT',
                                          'AVG_MIN_PRES', 'AVG_LF_PRES',])
    
    os.makedirs('tc_stats', exist_ok=True)
    # Save stats to csv
    stats.to_csv('tc_stats/{}_TC_stats_{}.csv'.format(name, region),index=False)
    
    return stats

def storm_statistics_sdd(storms, landfalls, nonlandfalls, name, region='GL',
                         debug=True, start_year=1979, end_year=2014, nan_value=0, scale=1, obs=False):
    ''' Creates a csv containing several climatological TC statistics.
        Parameters: 
        storms (pandas.DataFrame):
        landfalls (pandas.DataFrame:
        nonlandfalls (pandas.DataFrame)
        name (str):
        region (str):
        
    Returns:
        None
    '''
    # Number of ensemble members to scale by
    num_members = len(np.unique(storms.index.get_level_values('ENUM')))
    # Number of years based on start_year and end_year
    num_years = len(np.arange(start_year, end_year + 1, 1))
    
    storm_count       = 0
    landfall_count    = 0
    nonlandfall_count = 0

    if obs:
        scale = 1
        # Subset storms by time
        storms = storms[storms.index.get_level_values('SID').str[0:4].astype('int').isin(np.arange(start_year, end_year + 1, 1))]
        landfalls = landfalls[landfalls.index.get_level_values('SID').str[0:4].astype('int').isin(np.arange(start_year, end_year + 1, 1))]
        nonlandfalls = nonlandfalls[nonlandfalls.index.get_level_values('SID').str[0:4].astype('int').isin(np.arange(start_year, end_year + 1, 1))]
        
    storms       = subset_storms(region, storms)
    landfalls    = subset_storms(region, landfalls)
    nonlandfalls = subset_storms(region, nonlandfalls)

    for enum in np.unique(storms.index.get_level_values('ENUM')):
        ens_storms    = storms.xs(enum, level='ENUM', drop_level=False)
        ens_landfalls = landfalls.xs(enum, level='ENUM', drop_level=False)
        
        storm_count    += len(np.unique(ens_storms.index.get_level_values('SID')))
        landfall_count += len(np.unique(ens_landfalls.index.get_level_values('SID')))
        
    storm_count    = storm_count / num_members
    landfall_count = landfall_count / num_members
    
    # Percentage of landfalling/nonlandfalling TCs
    landfall_percent    = round((landfall_count / storm_count)*100, 1)
    nonlandfall_percent = round(100 - landfall_percent, 1)
    
    PACE = []
    ACE = []
    LMI = []
    min_pres = []

   # Coefficients for PACE calculation from Zarzycki et al. 2021
    a1, a2, a3 = -5.46649378e-05, 1.15460199e+00, 3.38263470e+01
    
    # Find timestep in data to determine how many rows to skip for calculation
    if obs:
        dt = pd.to_datetime(storms.ISO_TIME[3]).hour - pd.to_datetime(storms.ISO_TIME[2]).hour
        print(dt)
    else:
        dt = storms.HOUR[1] - storms.HOUR[0]

    skip = int(6 / dt)
    
    # Subset storm dataset by ensemble member
    num_members = 0
    for enum in np.unique(storms.index.get_level_values('ENUM')):
        ens_df = storms.xs(enum, level='ENUM', drop_level=False)
        num_members += 1
        for sid in np.unique(ens_df.index.get_level_values('SID')):
            storm_df = ens_df.xs(sid, level='SID')
            storm_df = storm_df[::skip]
            
            # If there is missing pressure information, calculate ACE instead
            if (storm_df.PRES.values == nan_value).any() == True:
                storm_wind_kts = storm_df.WIND.values * 1.944
                storm_ACE = (10**-4) * np.nansum((storm_wind_kts)**2)
                PACE.append(storm_ACE)
                ACE.append(storm_ACE)
                continue
                
            min_pres.append(np.nanmin(storm_df.PRES))
            LMI.append(np.abs((storm_df[storm_df.PRES==np.min(storm_df.PRES)]).LAT.values[0]))
            # Terms for PACE equation
            delta_p = 1010-storm_df.PRES.values
            a1_term = a1*(delta_p**2)
            a2_term = a2*(delta_p)
            
            # Calculating storm PACE
            storm_PACE = (10**-4) * np.nansum((a1_term + a2_term + a3)**2)
            
            storm_wind_kts = storm_df.WIND.values * 1.944
            storm_ACE = (10**-4) * np.nansum((storm_wind_kts)**2)
            
            PACE.append(storm_PACE)
            ACE.append(storm_ACE)
     
    avg_LMI = np.round(np.mean(LMI),1)
    avg_annual_PACE = np.round(((np.nansum(PACE) / num_years) * scale) / num_members, 1)
    avg_annual_ACE = np.round(((np.nansum(ACE) / num_years) * scale) / num_members, 1)
    avg_min_pres = np.round(np.nanmean(min_pres),1)
     
    lf_pres = []
    
    for enum in np.unique(landfalls.index.get_level_values('ENUM')):
        ens_df = landfalls.xs(enum, level='ENUM', drop_level=False)
        for i in range(len(ens_df)):
            landfall_df = ens_df.iloc[i]
            if landfall_df.PRES > 850:
                lf_pres.append(landfall_df.PRES)
    
    avg_lf_pres = np.round(np.nanmean(lf_pres),1)
    
    if debug:
        print('======={}======='.format(name))
        print('% Landfalling TC: {}\n% Non-landfalling TC: {}\nAverage lattitude of lifetime maximum intensity: {}\nAverage annual PACE: {}\nAverage annual ACE: {}\nAverage landfall pressure: {}\nAverage minimum pressure: {}'.format(landfall_percent,nonlandfall_percent,avg_LMI,avg_annual_PACE,avg_annual_ACE,avg_lf_pres,avg_min_pres))
    stats = [name,landfall_percent,nonlandfall_percent,avg_LMI,avg_annual_PACE,avg_annual_ACE,avg_lf_pres,avg_min_pres]
    stats = pd.DataFrame([stats],columns=['NAME','LF_PERCENT','NLF_PERCENT','AVG_LMI','AVG_PACE_YR','AVG_ACE_YR','AVG_LF_PRES','AVG_MIN_PRES'])

    os.makedirs('tc_stats', exist_ok=True)
    stats.to_csv('tc_stats/sdd_{}_TC_stats_{}.csv'.format(name,region),index=False)
    
    return stats

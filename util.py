import numpy as np
import xarray as xr
import pandas as pd
import math

def region_bounds(region,longitudes,latitudes):
    ''' Determines whether a storm genesis point is within a specified geographic region.
        Parameters: 
            region (str): oceanic domain that defines region bounds (options = 'GL','NA','EP','WP','IO','OC').
            lon (int arr): longitude(s) of storm genesis point(s) to be checked.
            lat (int arr): latitude(s) of storm genesis point(s) to be checked.
            
        Returns:
            in_region (bool): True if point is in region, False if point is not in region.
    '''
    # Convert single lon/lat values to lists so that list comprehension code doesn't break
    if isinstance(longitudes,np.float64) and isinstance(latitudes,np.float64):
        longitudes = [longitudes]
        latitudes  = [latitudes]
    
    try:
        # Global
        if region == "GL":
            return[True for lon,lat in zip(longitudes,latitudes)]

        # North Atlantic
        if region == "NA":
            ('In NA')
            return [(True if -100 <= lon <= 0 and 0 <= lat <= 60 else False) for lon,lat in zip(longitudes, latitudes)]

        # East Pacific
        if region == "EP":
            return [(True if -180 <= lon <= -70 and 0 <= lat <= 60 else False) for lon,lat in zip(longitudes, latitudes)]

        # West Pacific
        if region == "WP":
            return [(True if 100 <= lon <= 180 and 0 <= lat <= 60 else False) for lon,lat in zip(longitudes, latitudes)]

        # Northern Indian Ocean
        if region == "NIO":
            return [(True if 30 <= lon <= 100 and 0 <= lat <= 50 else False) for lon,lat in zip(longitudes, latitudes)]

        # Southern Indian Ocean
        if region == "SIO":
            return [(True if -20 <= lon <= 100 and -45 <= lat <= 0 else False) for lon,lat in zip(longitudes, latitudes)]

        # Oceania
        if region == "OC":
            return [(True if 100 <= lon <= 180 and -45 <= lat <= 0 else False) for lon,lat in zip(longitudes, latitudes)]

    except:
        print("Invalid region specified. Please select a region from the following options: 'GL', 'NA', 'EP', 'WP', 'NIO', 'SIO', 'OC'")
    
    
def subset_storms(region,storms):
    in_region = region_bounds(region,storms.GEN_LON.values,storms.GEN_LAT.values)
    
    return storms.loc[in_region]
    
    
    
def storm_statistics(storms,landfalls,nonlandfalls,name,region='GL',start_year=1979,end_year=2014,debug=True):
    ''' Creates a csv containing several climatological TC statistics.
        Parameters: 
        storms (pandas.DataFrame):
        landfalls (pandas.DataFrame:
        nonlandfalls (pandas.DataFrame)
        name (str):
        region (str):
        start_year (int)
        end_year (int):
        
    Returns:
        None
    '''
    # Number of ensemble members to scale by
    try:
        num_members = len(np.unique(storms.index.get_level_values('ENUM')))
        obs = False
    except:
        num_members = 1
        obs = True
    
    # Length of dataset (in years)
    years = np.unique(storms.index.get_level_values('ISO_TIME').year.values)
    
    # Truncate dataset to desired length
    subset_years = np.where(np.logical_and(start_year <= years, years <= end_year))[0]
    length = len(subset_years)
    
    storms = storms[(storms.index.get_level_values('ISO_TIME').year >= start_year) & (storms.index.get_level_values('ISO_TIME').year <= end_year)]
    landfalls = landfalls[(landfalls.index.get_level_values('ISO_TIME').year >= start_year) & (landfalls.index.get_level_values('ISO_TIME').year <= end_year)]
    nonlandfalls = nonlandfalls[(nonlandfalls.index.get_level_values('ISO_TIME').year >= start_year) & (nonlandfalls.index.get_level_values('ISO_TIME').year <= end_year)]
    
    # Subset storms by basin (if applicable)
    storms_in_region       = subset_storms(region,storms)
    landfalls_in_region    = subset_storms(region,landfalls)
    nonlandfalls_in_region = subset_storms(region,nonlandfalls)
    
    # Average annual TC count
    storm_count = math.ceil(len(np.unique(storms_in_region.index.get_level_values('SID')))/num_members)
    storm_count_annual = np.round(storm_count/length,decimals=1)

    # Average annual landfall count
    landfall_count = math.ceil(len(np.unique(landfalls_in_region.index.get_level_values('SID')))/num_members)
    landfall_count_annual = np.round(landfall_count/length,decimals=1)
    
    # Average annual nonlandfall count
    nonlandfall_count = math.ceil(len(np.unique(nonlandfalls_in_region.index.get_level_values('SID')))/num_members)
    nonlandfall_count_annual = np.round(nonlandfall_count/length,decimals=1)
    
    # Percentage of landfalling/nonlandfalling TCs
    landfall_percent = np.round((landfall_count/storm_count)*100,1)
    nonlandfall_percent = np.round((nonlandfall_count/storm_count)*100,1)
    
    # Average annual number of TC days
    if obs:
        dates = np.unique(storms_in_region.index.get_level_values('ISO_TIME').date)
        dates = np.asarray(dates)
        
        tc_days = np.round(len(dates)/length,1)
        
    else:
        tc_days_ens = []
        for enum in np.unique(storms_in_region.index.get_level_values('ENUM')):
            storms_ens = storms_in_region.xs(enum,level='ENUM')                   
            
            dates_ens = np.unique(storms_ens.index.get_level_values('ISO_TIME').date)
            dates_ens = np.asarray(dates_ens)
            
            tc_days_ens.append(len(dates_ens)/length)

        tc_days = np.round(np.mean(tc_days_ens),1)

    # Average storm lifetime before landfall (in days) and average latitude of maximum intensity
    lifetimes = []
    lmis = []
    for sid in (np.unique(landfalls.index.get_level_values('SID'))):
        storm_df = storms.xs(sid,level='SID')
        
        # don't include storms that have missing pressure data in the lmi statistic
        try:
            lmi = np.abs((storm_df[storm_df.PRES==np.min(storm_df.PRES.values)]).LAT.values[0])
            lmis.append(lmi)
            
        except:
            pass
            
        dates = np.unique(storm_df.index.get_level_values('ISO_TIME').date)
        lifetimes.append(len(dates))
    
    med_lifetime = np.median(lifetimes)
    avg_lifetime = np.round(np.mean(lifetimes),decimals=1)
    avg_lmi = np.round(np.mean(lmis),decimals=1)

    if debug:
        print('======={}======='.format(name))
        print('Total TC count (averaged by # of ensemble members): {}\nLandfalling TC count (averaged by # of ensemble members): {}\nNon-landfalling TC count (averaged by # of ensemble members): {}\nAverage annual TC count: {}\nAverage annual landfall count: {}\n% Landfalling TC: {}\n% Non-landfalling TC: {}\nAverage annual TC days: {}\nAverage storm lifetime before landfall (days): {}\nMedian storm lifetime before landfall (days): {}\nAverage lattitude of lifetime maximum intensity: {}'.format(storm_count,landfall_count,nonlandfall_count,storm_count_annual,landfall_count_annual,
                      landfall_percent,nonlandfall_percent,tc_days,avg_lifetime,med_lifetime,avg_lmi))
    
    stats = [name,storm_count,landfall_count,nonlandfall_count,storm_count_annual,landfall_count_annual,
             landfall_percent,nonlandfall_percent,tc_days,avg_lifetime,med_lifetime,avg_lmi]
    stats = pd.DataFrame([stats],columns=['NAME','STORM_COUNT','LF_COUNT','NLF_COUNT','STORM_COUNT_YR','LF_COUNT_YR',
                                          'LF_PERCENT','NLF_PERCENT','TC_DAYS_YR','AVG_LIFE','MED_LIFE','AVG_LMI'])
    
    stats.to_csv('{}_TC_stats_{}.csv'.format(name,region),index=False)
    
    return stats

import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import griddata

def interp_location_landfrac(storm_df,landfrac_points,landfrac_values,frequency,interval):
    ''' Interpolates track dataset to a specified time frequency.
        Parameters: 
            storm_df (pandas.DataFrame): raw storm track data.
            landfrac_points (pandas.DataFrame): grid of land fraction points.
            landfrac_values (pandas.DataFrame): grid of land fraction values.
            frequency (str): time frequency for interpolation.
            interval (int): hourly interval of input data
            
        Returns:
            interped_df (pandas.DataFrame): interpolated storm track data.
    '''
    storm_times = storm_df.index.get_level_values('ISO_TIME')
    
    # Final check to ensure times are in datetime format
    if type(storm_times[0]) == str:
        storm_times = pd.to_datetime(storm_times, format='%Y-%m-%d %H:%M:%S')

    # Set desired time range (according to desired frequency)
    storm_time_hourly = pd.date_range(storm_times[0],storm_times[-1],freq=frequency)
    temp_df           = storm_df.copy()
    temp_df           = temp_df.reset_index()

    # Create new ISO_TIME column the length of storm_time_hourly, input existing times at appropriate indices
    for time in storm_time_hourly:
        length = len(temp_df)
        if time.strftime("%Y-%m-%d %H:%M:%S") not in storm_times:
            temp_df.loc[length,'ISO_TIME'] = time

    temp_df       = temp_df.set_index(['ISO_TIME'])
    temp_df.index = pd.to_datetime(temp_df.index)
    
    temp_df       = temp_df.sort_index()

    # Linearly interpolate DataFrame
    diff = temp_df['LON'].diff(periods=interval)
    # For storm tracks that cross over the dateline, do some funky stuff so that pandas doesn't explode
    if (abs(diff.values) > 180).any():
        # Make any lon values above 0 negative (keeps values adjacent to each other so interpolation can be done correctly)
        temp_df['LON']     = np.where(temp_df.LON.values < 0, temp_df.LON.values, temp_df.LON.values - 360)
        interped_df        = temp_df.interpolate(axis='index',method='linear')
        # Convert lon values back into a -180/180 format
        interped_df['LON'] = np.where(interped_df.LON.values > -180, interped_df.LON.values, interped_df.LON.values + 360)

    else:
        interped_df   = temp_df.interpolate(axis='index',method='linear')

    temp_points   = list(zip(interped_df['LON'].values,interped_df['LAT'].values)) # Put points together as a list 

    temp_landfrac = griddata(landfrac_points,landfrac_values,temp_points,method='nearest') # Interpolate landfrac
    
    # Toss any storms that start over land
    if temp_landfrac[0] >= 0.5:
        return None

    interped_df['LANDFRAC'] = temp_landfrac
    
    return interped_df


def create_storm_list(track_data,landfrac_points,landfrac_values,frequency,interval,calendar):
    ''' Creates complete list of storm tracks with all associated information.
        Parameters:
            track_data (pandas.DataFrame): complete, uninterpolated storm track data.
            landfrac_points (pandas.DataFrame): grid of land fraction points.
            landfrac_values (pandas.DataFrame): grid of land fraction values
            frequency (str): time frequency for interpolation.
            
        Returns:
            storms (pandas.DataFrame): complete, interpolated storm track data.
            
    '''
    print("    Generating storm list...")
    
    # Array of pandas dataframes
    storm_sid_land_list = []                                

    print("        Interpolating data to {}...".format(frequency))
    
    for enum in np.unique(track_data.index.get_level_values('ENUM')[:]): # Loop over ENUM
        ens_member = track_data.xs(enum,level='ENUM').copy()
        for sid in np.unique(ens_member.index.get_level_values('SID'))[:]:   # Loop over SID
            storm_sid        = ens_member.xs(sid,level='SID').copy()  # Select the storm whose id is sid

            storm_sid_interp = interp_location_landfrac(storm_sid,landfrac_points,landfrac_values,frequency,interval)

            # None is returned if storm starts over land. Storm is not included in the dataframe
            if storm_sid_interp is None:
                continue

            storm_sid_interp['SID']            = sid
            storm_sid_interp['ENUM']           = enum
            storm_sid_interp = storm_sid_interp.set_index(['ENUM','SID'],append=True)
            storm_sid_interp = storm_sid_interp.reorder_levels(['ENUM','SID','ISO_TIME'])
            storm_sid_land_list.append(storm_sid_interp)
        

    storms = pd.concat(storm_sid_land_list,axis=0) # Combine all storms into one dataframe again
    
    # Round any land fraction values >0.5 to 1 (land) and <= 0.5 to 0 (ocean)
    storms.loc[storms['LANDFRAC']<=.5,'LANDFRAC'] = 0
    storms.loc[storms['LANDFRAC']>.5,'LANDFRAC'] = 1
    
    print('        Storm list generated!')

    return storms
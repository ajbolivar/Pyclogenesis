import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import griddata

# Interpolate track dataset to a specified time frequency.
def interp_location_landfrac(storm_df,landfrac_points,landfrac_values,frequency,interval):
    storm_times = storm_df.index.get_level_values('time')
    
    # Final check to ensure times are in datetime format
    if type(storm_times[0]) == str:
        storm_times = pd.to_datetime(storm_times, format='%Y-%m-%d %H:%M:%S')

    # Set desired time range (according to desired frequency)
    storm_time_hourly = pd.date_range(storm_times[0], storm_times[-1], freq=frequency)
    temp_df           = storm_df.copy()
    temp_df           = temp_df.reset_index()

    # Create new ISO_TIME column the length of storm_time_hourly, input existing times at appropriate indices
    for time in storm_time_hourly:
        length = len(temp_df)
        if time.strftime("%Y-%m-%d %H:%M:%S") not in storm_times:
            temp_df.loc[length, 'time'] = time

    temp_df       = temp_df.set_index('time')
    temp_df.index = pd.to_datetime(temp_df.index).sort_index()

    # Linearly interpolate DataFrame
    diff = temp_df['lon'].diff(periods=interval)
    # For storm tracks that cross over the dateline, do some funky stuff so that pandas doesn't explode
    if (abs(diff.values) > 180).any():
        # Make any lon values above 0 negative (keeps values adjacent to each other so interpolation can be done correctly)
        temp_df['lon']     = np.where(temp_df.lon.values < 0, temp_df.lon.values, temp_df.lon.values - 360)
        interped_df        = temp_df.interpolate(axis='index', method='linear')
        # Convert lon values back into a -180/180 format
        interped_df['lon'] = np.where(interped_df.lon.values > -180, interped_df.lon.values, interped_df.lon.values + 360)

    else:
        interped_df   = temp_df.interpolate(axis='index', method='linear')

    temp_points   = list(zip(interped_df['lon'].values, interped_df['lat'].values)) # Put points together as a list 

    temp_landfrac = griddata(landfrac_points, landfrac_values, temp_points, method='nearest') # Interpolate landfrac
    
    # Toss any storms that start over land
    if temp_landfrac[0] >= 0.5:
        return None

    interped_df['landfrac'] = temp_landfrac
    
    return interped_df

# Create complete list of storm tracks with all associated information.
def create_storm_list(track_data,landfrac_points,landfrac_values,frequency,interval,calendar):
    print("    Generating storm list...")
    
    # Array of pandas dataframes
    storm_sid_land_list = []                                

    print("        Interpolating data to {}...".format(frequency))
    
    for enum in np.unique(track_data.index.get_level_values('enum')[:]): # Loop over ENUM
        ens_member = track_data.xs(enum, level='enum').copy()
        for sid in np.unique(ens_member.index.get_level_values('sid'))[:]:   # Loop over SID
            storm_sid        = ens_member.xs(sid, level='sid').copy()  # Select the storm whose id is sid

            storm_sid_interp = interp_location_landfrac(storm_sid, landfrac_points, landfrac_values, frequency, interval)

            # None is returned if storm starts over land. Storm is not included in the dataframe
            if storm_sid_interp is None:
                continue

            storm_sid_interp['sid']  = sid
            storm_sid_interp['enum'] = enum
            storm_sid_interp = storm_sid_interp.set_index(['enum', 'sid'], append=True)
            storm_sid_interp = storm_sid_interp.reorder_levels(['enum', 'sid', 'time'])
            storm_sid_land_list.append(storm_sid_interp)
        

    storms = pd.concat(storm_sid_land_list,axis=0) # Combine all storms into one dataframe again
    
    # Round any land fraction values >0.5 to 1 (land) and <= 0.5 to 0 (ocean)
    storms.loc[storms['landfrac']<=.5, 'landfrac'] = 0
    storms.loc[storms['landfrac']>.5, 'landfrac'] = 1
    
    print('        Storm list generated!')

    return storms
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import griddata
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.io.shapereader as shpreader
import os
import shapely.geometry as shp_geom
import random
from Pyclogenesis import interpolate
from Pyclogenesis import util

# A master function that calls all other functions in Pyclogenesis to return a complete list of storm track data and landfall locations.
def tc_list(tracks,data_source,interval,flip_lon=False,calendar='standard',
            frequency='1H',hours_over_water=12,start_year=1979,no_ET=True):
    track_data, landfrac_points, landfrac_values = load_track_data(tracks=tracks,data_source=data_source,flip_lon=flip_lon,
                                                                   calendar=calendar,start_year=start_year,no_ET=no_ET)
    storms = interpolate.create_storm_list(track_data=track_data,landfrac_points=landfrac_points,
                                           landfrac_values=landfrac_values,frequency=frequency,
                                           interval=interval,calendar=calendar)
    
    landfalls = find_landfalls(storms=storms,hours_over_water=hours_over_water)
    
    return storms, landfalls


# Formats data to work with Pyclogenesis
def load_track_data(tracks,data_source,flip_lon,calendar,start_year,no_ET):
    # Load land fraction data
    # MODIFY FILEPATH ACCORDING TO WHERE YOU STORE YOUR DATA
    landfrac_dat = '/glade/work/abolivar/Pyclogenesis_data/landfrac_data/USGS_gtopo30_0.23x0.31_remap_c180612_PHIS_LANDFRAC.nc'
    print("Loading track data located at '{}'...".format(tracks))

    landfrac = xr.open_dataset(landfrac_dat)
    landfrac = landfrac.assign_coords(lon = (((landfrac.lon + 180) % 360) - 180))
    landfrac = landfrac.sortby('lon')

    # Observational dataset
    if data_source == 'obs':
        # Read file
        track_data = pd.read_csv(tracks, usecols = ['SID','ISO_TIME','NATURE','LAT','LON','WMO_WIND','WMO_PRES'], 
                                 index_col=['SID','ISO_TIME'], parse_dates=True, low_memory=False,
                                 names=['sid', 'time', 'nature', 'lat', 'lon', 'wind', 'pressure'])[1:]
        
        if no_ET:
            # Filter out extratropical storms, disturbances, and unclassified storms
            track_data = track_data[track_data.nature.isin(['TS','SS','DS'])]
            
        # Drop column with storm classifications    
        track_data = track_data.drop(columns='nature')
        
        track_data['lat']   = track_data['lat'].copy().astype(float)
        track_data['lon']   = track_data['lon'].copy().astype(float)
        track_data_wind     = track_data['wind']
        track_data_pres     = track_data['pressure']
        
        # Replace missing wind and pressure values with NaNs
        track_data_wind_nan = track_data_wind.where(track_data_wind!=' ',np.nan).dropna().astype(float)
        track_data_pres_nan = track_data_pres.where(track_data_pres!=' ',np.nan).dropna().astype(float)
        
        track_data = pd.concat([track_data[['lat','lon']], track_data_wind_nan,track_data_pres_nan], axis=1)
        
        # Toss out any times that do not fall on XX:00
        hour_check = pd.to_datetime(track_data.index.get_level_values('time')).hour.isin(np.arange(0,24,3))
        minute_check = pd.to_datetime(track_data.index.get_level_values('time')).minute == 0
        second_check = pd.to_datetime(track_data.index.get_level_values('time')).second == 0

        time_check = (hour_check) & (minute_check) & (second_check)
        track_data = track_data.loc[time_check]
        
        # Truncate dataset to specified start time
        start = pd.to_datetime(track_data.index.get_level_values('time')).year >= start_year
        track_data = track_data.loc[start]
        
        gen_lat = []
        gen_lon = []
        enum = []
        
        # Generate column of genesis lon/lat information
        for sid in track_data.index.get_level_values('sid'):
            storm_df = track_data.xs(sid, level='sid')
            gen_lat.append(storm_df.lat.values[0])
            gen_lon.append(storm_df.lon.values[0])
            enum.append('obs')
        
        track_data['gen_lat'] = gen_lat
        track_data['gen_lon'] = gen_lon
        track_data['enum']    = enum
        
        track_data['lon']     = np.where(track_data.lon.values < 180, track_data.lon.values, track_data.lon.values - 360)
        track_data['lon']     = np.where(track_data.lon.values > -180, track_data.lon.values, track_data.lon.values + 360)
        
        track_data = track_data.reset_index()
        track_data = track_data.set_index(['enum', 'sid', 'time'])
        
    # Model/reanalysis datasets
    elif data_source == 'model' or data_source == 'reanalysis':
        track_data = pd.DataFrame()
        
        start_idx = 0
        for track_file in os.listdir(tracks):
            if 'ipynb' in track_file:
                continue
                
            # Parse file into workable data
            td = open(tracks + track_file)
            lines = td.readlines()
            # Split lines by tab and remove new line characters
            lines_split = [l.replace('\n','').split('\t')for l in lines]
            # Find all start line indexes and append last line too
            lines_start = [i for i in range(0,len(lines_split)) if lines_split[i][0]=='start'] + [len(lines_split)] 
            storm_length = list(np.diff(lines_start) - 1) # Minus one because start line would be included if not

            sid       = '{}{:02d}{:02d}{:02d}{}{}'
            
            # Go to lines_start[:-1] because last line is not a start line
            gen_lons  = [[(float(lines_split[start+1][3]) + 180) % 360 - 180] * length for start,length in zip(lines_start[:-1],storm_length)]
            gen_lats  = [[float(lines_split[start+1][4])] * length for start,length in zip(lines_start[:-1],storm_length)]
            
            # Create strings for gen lons and lats for storm ID
            gen_lon_str = [str(np.abs(int(gen_lon[0])))+'W' if (int(gen_lon[0]) < 0) else str(np.abs(int(gen_lon[0])))+'E' if (int(gen_lon[0]) > 0) else gen_lon[0] for gen_lon in gen_lons]
            gen_lat_str = [str(np.abs(int(gen_lat[0])))+'S' if (int(gen_lat[0]) < 0) else str(np.abs(int(gen_lat[0])))+'N' if (int(gen_lat[0]) > 0) else gen_lat[0] for gen_lat in gen_lats]
            
            # Generate storm IDs by concatenating spatial and temporal information of storm genesis
            storm_ids = [[sid.format(*list(np.array(lines_split[start][2:]).astype(int))+[gen_lat]+[gen_lon])] * length for start,length,gen_lat,gen_lon in zip(lines_start[:-1],storm_length,gen_lat_str,gen_lon_str)]
            
            # Adding ensemble number information to dataframe
            if data_source == 'model':
                ensemble = [[track_file.split('_')[-3]] * length for length in storm_length]
                #if track_file[-32] != 'r': ensemble = [[track_file[-33:-24]] * length for length in storm_length]
                #else: ensemble = [[track_file[-32:-24]] * length for length in storm_length]
            else: ensemble = [['reanalysis'] * length for length in storm_length]

   
            # Create DataFrame from track file, skip header rows (those beginning with start)
            member_data = pd.read_table(tracks+track_file, skiprows=lines_start[:-1], usecols=([3,4,5,6,8,9,10,11]),
                                        delimiter = '\t', header=None, 
                                        names=['lon','lat','pressure','wind','YYYY','MM','DD','HH'])
            
            member_data['sid']      = flatten(storm_ids)
            member_data['gen_lon']  = flatten(gen_lons)
            member_data['gen_lat']  = flatten(gen_lats)
            member_data['pressure'] = (member_data['pressure'].values)/100
            member_data['ENUM']     = flatten(ensemble)
            member_data             = member_data.set_index(['sid']).sort_index().reset_index()
                     
            y = [int(member_data.loc[time,'YYYY']) for time in member_data.index]
            m = [int(member_data.loc[time,'MM']) for time in member_data.index]
            d = [int(member_data.loc[time,'DD']) for time in member_data.index]
            h = [int(member_data.loc[time,'HH']) for time in member_data.index]
            
            if calendar == 'standard':
                time = ['{}{:02d}{:02d}{:02d}'.format(year, month, day, hour) for year,month,day,hour in zip(y, m, d, h)]
                datetime = [pd.to_datetime(t, format='%Y%m%d%H') for t in time]

            if calendar == '360-day':
                time = ['{}{:02d}-{:02d}{:02d}'.format(year, month, day, hour) for year, month, day, hour in zip(y, m, d, h)]
                # Create 365-day calendar to map indices
                std_calendar = pd.date_range(start='1/1/1999', end='12/31/1999')
                calendar_365 = [str(std_calendar[i])[5:10] for i in range(len(std_calendar))]

                # Map 360-day calendar dates to 365-day calendar date indices through the following formula: 30(m - 1) + d - 1
                index = [(((month - 1) * 30) + day - 1) for month, day in zip(m, d)]
                # Format month/day of 360-day calendar the same way as the 365-day calendar for ease of comparison
                date  = [(str('%02d' % month) + '-' + str('%02d' % day)) for month,day in zip(m, d)]
                # Replace 360-day calendar dates with appropriate 365-day calendar dates according to the above indices
                new_time = [t.replace(d, calendar_365[i]) for t, d, i in zip(time, date, index)]
                # Populate datetime with new dates
                datetime = [pd.to_datetime(newt, format='%Y%m-%d%H') for newt in new_time]

            member_data['time'] = datetime
            member_data = member_data.drop(columns=['YYYY', 'MM', 'DD', 'HH'])

            # Truncate dataset to specified start time
            start = member_data.datetime.dt.year >= start_year
            member_data = member_data.loc[start]
            
            track_data = pd.concat([track_data,member_data])
            # track_data = track_data.set_index(['time'],append=True)

        track_data = track_data.set_index(['enum', 'sid', 'time'],append=True)
        track_data = track_data.droplevel(0, axis=0)
            
    # Convert lon/lat from 0/360 to -180/180 if flip_lon is set to True
    if flip_lon:
        track_data['lon'] = (track_data.lon.values + 180) % 360 - 180
        track_data['gen_lon'] = (track_data.gen_lon.values + 180) % 360 - 180                                           

    # Create and fit landfrac mesh to lon/lat grid
    lon_landfrac = landfrac.lon.values
    lat_landfrac = landfrac.lat.values
    lon_landfrac_mesh, lat_landfrac_mesh = np.meshgrid(lon_landfrac, lat_landfrac)

    landfrac_points = (lon_landfrac_mesh.flatten(), lat_landfrac_mesh.flatten())
    landfrac_values = landfrac.LANDFRAC.values.flatten()

    track_data = track_data.sort_index()
    print("    Track data loaded!")
    
    return track_data, landfrac_points, landfrac_values



def flatten(l):
    return [item for sublist in l for item in sublist]     


# Create a list of landfalling and non-landfalling storms.
def find_landfalls(storms, hours_over_water=12, enums=None, region='GL', landfrac_thresh=0.5, lat_thresh=45.0):
    print("    Generating landfalling/non-landfalling storm lists...")

    landings_list = []
    nonlandings_list = []
    
    # If enums is not user-specified, use all enums in the storm DataFrame
    if enums == None:
        enums = np.unique(storms.index.get_level_values('enum'))
    
    for enum in enums:
        enum_df = storms.xs(enum, level='enum',drop_level=False)
        sids = np.unique(enum_df.index.get_level_values('sid'))
        for sid in sids: # Iterate through SIDs
            storm_df = enum_df.xs(sid, level='sid', drop_level=False)
            
            extratropical = False
            landfall = False
            
            times = storm_df.index.get_level_values('time').values
            if (storm_df.landfrac > landfrac_thresh).any():
                for itime in range(len(times)): # For all times and time indices in storm_df...
                    if (storm_df.iloc[itime].landfrac > landfrac_thresh):  # Check if landfrac value is greater than 0.5...
                        # Check if it has been at least X many hours since last landfrac == 1, where X = hours_over water...
                        if itime >= hours_over_water:
                            if (storm_df.iloc[itime-hours_over_water:itime].landfrac <= landfrac_thresh).all():
                                storm_time = storm_df.iloc[itime]
                                lat = storm_time.lat
                                # Append landfall if it falls within the lon/lat bounds
                                if np.abs(lat) <= lat_thresh:
                                    landings_list.append(storm_df.iloc[[itime]])
                                    landfall = True
                                # Append as nonlandfall if it fails the above check
                                else:
                                    extratropical = True
                                    break

                        # Alternative check for storms that make landfall at less than X hours old, where X = hours_over water...
                        else:
                            if (itime > 0) and (storm_df.iloc[0:itime].landfrac <= 0.5).all():
                                landings_list.append(storm_df.iloc[[itime]])
                                
                if extratropical and not landfall: continue

    landfalls = pd.concat(landings_list, axis=0)
    
    print('        Landfalling/non-landfalling storm lists generated!')
    
    return landfalls
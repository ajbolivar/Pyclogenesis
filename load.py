import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import griddata
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.io.shapereader as shpreader
import os
import shapely.geometry as shp_geom
from Pyclogenesis import interpolate
from Pyclogenesis import util

def tc_list(tracks,data_source,interval,flip_lon=False,calendar='standard',
            frequency='1H',hours_over_water=12,start_year=1979):
    '''A master function that calls all other functions in Pyclogenesis to return a complete list of storm track data and landfall locations.
       Parameters: 
           tracks (str): filepath of the storm track data to be processed (file for obs, dir for model/reanalysis).
           data_source (str): specification of where the data is coming from (options: 'obs','model','reanalysis').
           flip_lon (bool): when True, converts longitude from (0 to 360) degrees to (-90 to 90) degrees (default = False).
           calendar (str): specification of model calendar type (options: 'standard', '360-day'; default = 'standard')
           frequency (str): desired time frequency for interpolation (default = '1H').
           interval (int): time interval (in hours) of input file data.
           hours_over_water (int): minimum hours that a landfalling storm must be over water before another landfall is counted (default = 12). 
           states (bool): when True, extra analysis is performed to categorize landfalls by U.S. state (default = False).
            
       Returns:
           track_dat (pandas.DataFrame): track data converted to a DataFrame.
           landfrac_points (pandas.DataFrame): grid of landfraction points according to model grid spacing.
           landfrac_values (pandas.DataFrame): land fraction values from source.
    '''
    
    track_data, landfrac_points, landfrac_values = load_track_data(tracks=tracks,data_source=data_source,
                                                                  flip_lon=flip_lon,calendar=calendar,start_year=start_year)
    storms = interpolate.create_storm_list(track_data=track_data,landfrac_points=landfrac_points,landfrac_values=landfrac_values,
                                           frequency=frequency,interval=interval,calendar=calendar)
    
    landfalls, nonlandfalls = find_landfalls(storms=storms,hours_over_water=hours_over_water)
    
    return storms, landfalls, nonlandfalls
    


def load_track_data(tracks,data_source,flip_lon,calendar,start_year):
    ''' Using a track data file and a specified data source, formats data to work with Pyclogenesis.
        Parameters: 
            tracks (str): filepath of the storm track data to be processed.
            data_source (str): specification of where the data is coming from (options: 'obs','model','reanalysis').
            flip_lon (bool): when True, converts longitude from (0 to 360) degrees to (-90 to 90) degrees (default = False).
            calendar (str): specification of model calendar type (options: 'standard')
            
        Returns:
            track_data (pandas.DataFrame): raw list of storm tracks
            landfrac_points (pandas.DataFrame): grid of landfrac points
            landfrac_values (pandas.DataFrame): grid of landfrac values
    '''
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
        track_data = pd.read_csv(tracks, usecols = ['SID','ISO_TIME','LAT','LON','WMO_WIND','WMO_PRES'], 
                                 index_col=['SID','ISO_TIME'],parse_dates=True,low_memory=False)[1:]
        
        track_data['LAT']   = track_data['LAT'].copy().astype(float)
        track_data['LON']   = track_data['LON'].copy().astype(float)
        track_data_WMO_WIND = track_data['WMO_WIND']
        track_data_WMO_PRES = track_data['WMO_PRES']
        
        track_data_WMO_PRES.name = 'PRES'
        track_data_WMO_WIND.name = 'WIND'
        
        # Replace missing wind and pressure values with NaNs
        track_data_WMO_WIND_nan = track_data_WMO_WIND.where(track_data_WMO_WIND!=' ',np.nan)
        track_data_WMO_PRES_nan = track_data_WMO_PRES.where(track_data_WMO_PRES!=' ',np.nan)
        
        # Remove missing data
        track_data_WMO_WIND_nan = track_data_WMO_WIND_nan.dropna() # remove missing data
        track_data_WMO_PRES_nan = track_data_WMO_PRES_nan.dropna()
        
        # Make remaining data into floats
        track_data_WMO_WIND_nan = track_data_WMO_WIND_nan.astype(float)
        track_data_WMO_PRES_nan = track_data_WMO_PRES_nan.astype(float)
        
        track_data = pd.concat([track_data[['LAT','LON']],track_data_WMO_WIND_nan,track_data_WMO_PRES_nan],axis=1)
        
        # Toss out any times that do not fall on XX:00
        hour_check = pd.to_datetime(track_data.index.get_level_values('ISO_TIME')).hour.isin(np.arange(0,24,3))
        minute_check = pd.to_datetime(track_data.index.get_level_values('ISO_TIME')).minute == 0
        second_check = pd.to_datetime(track_data.index.get_level_values('ISO_TIME')).second == 0

        time_check = (hour_check) & (minute_check) & (second_check)
        track_data = track_data.loc[time_check]
        
        # Truncate dataset to specified start time
        start = pd.to_datetime(track_data.index.get_level_values('ISO_TIME')).year >= start_year
        track_data = track_data.loc[start]
        
        gen_lat = []
        gen_lon = []
        enum = []
        
        # Generate column of genesis lon/lat information
        for sid in track_data.index.get_level_values('SID'):
            storm_df = track_data.xs(sid,level='SID')
            gen_lat.append(storm_df.LAT.values[0])
            gen_lon.append(storm_df.LON.values[0])
            enum.append('OBS')
        
        track_data['GEN_LAT'] = gen_lat
        track_data['GEN_LON'] = gen_lon
        track_data['ENUM'] = enum
        
        track_data['LON']     = np.where(track_data.LON.values < 180, track_data.LON.values, track_data.LON.values - 360)
        track_data['LON']     = np.where(track_data.LON.values > -180, track_data.LON.values, track_data.LON.values + 360)
        
        track_data = track_data.reset_index()
        track_data = track_data.set_index(['ENUM','SID','ISO_TIME'])
        
    # Model/reanalysis datasets
    elif data_source == 'model' or data_source == 'reanalysis':
        # Model/reanalysis datasets
        # Create unique 4-character strings for all storm IDs
        alphabets = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 
                     'h', 'i', 'j', 'k', 'l', 'm', 'n', 
                     'o', 'p', 'q', 'r', 's', 't', 'u', 
                     'v', 'w', 'x', 'y', 'z']

        indices = []
        for alpha1 in alphabets:
            for alpha2 in alphabets:
                for alpha3 in alphabets:
                    for alpha4 in alphabets:
                        indices.append(alpha1+alpha2+alpha3+alpha4)

        track_data = pd.DataFrame()

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

            storm_length = np.diff(lines_start) - 1 # Minus one because start line would be included if not

            sid       = '{}{:02d}{:02d}{:02d}_{}'
            storm_ids = [[sid.format(*list(np.array(lines_split[start][2:]).astype(int))+[ix])] * length 
                             for start,length,ix in zip(lines_start[:-1],storm_length,indices[:len(lines_start)])]
            gen_lons  = [[float(lines_split[start+1][3])] * length for start,length in zip(lines_start[:-1],storm_length)]
            gen_lats  = [[float(lines_split[start+1][4])] * length for start,length in zip(lines_start[:-1],storm_length)]
            
            # Adding ensemble ID information to dataframe
            if data_source == 'model':
                if track_file[-32] != 'r': ensemble = [[track_file[-33:-24]] * length for length in storm_length]
                else: ensemble = [[track_file[-32:-24]] * length for length in storm_length]
            else: ensemble = [['REANALYSIS'] * length for length in storm_length]


            # Create DataFrame from track file, skip header rows (those beginning with start)
            member_data = pd.read_table(tracks+track_file, skiprows=lines_start[:-1], usecols=([3,4,5,6,8,9,10,11]),
                                        delimiter = '\t', header=None, 
                                        names=['LON','LAT','PRES','WIND','YYYY','MM','DD','HH'])
            
            member_data['SID']     = flatten(storm_ids)
            member_data['GEN_LON'] = flatten(gen_lons)
            member_data['GEN_LAT'] = flatten(gen_lats)
            member_data['PRES']    = (member_data['PRES'].values)/100
            member_data['ENUM']    = flatten(ensemble)
            member_data            = member_data.set_index(['SID']).sort_index()
                    
            storms = member_data.reset_index()

            # Failsafe in case storms is a series object instead of a dataframe
            if isinstance(storms,pd.Series):
                storms = storms.to_frame().T
                    
            y = [int(storms.loc[time,'YYYY']) for time in storms.index]
            m = [int(storms.loc[time,'MM']) for time in storms.index]
            d = [int(storms.loc[time,'DD']) for time in storms.index]
            h = [int(storms.loc[time,'HH']) for time in storms.index]
            
            if calendar == 'standard':
                time = ['{}{:02d}{:02d}{:02d}'.format(year,month,day,hour) for year,month,day,hour in zip(y,m,d,h)]
                iso_time = [pd.to_datetime(t, format='%Y%m%d%H') for t in time]

            if calendar == '360-day':
                time = ['{}{:02d}-{:02d}{:02d}'.format(year,month,day,hour) for year,month,day,hour in zip(y,m,d,h)]
                # Create 365-day calendar to map indices
                std_calendar = pd.date_range(start='1/1/1999',end='12/31/1999')
                calendar_365 = [str(std_calendar[i])[5:10] for i in range(len(std_calendar))]

                # Map 360-day calendar dates to 365-day calendar date indices through the following formula: 30(m - 1) + d - 1
                index = [(((month - 1) * 30) + day - 1) for month,day in zip(m,d)]
                # Format month/day of 360-day calendar the same way as the 365-day calendar for ease of comparison
                date  = [(str('%02d' % month) + '-' + str('%02d' % day)) for month,day in zip(m,d)]
                # Replace 360-day calendar dates with appropriate 365-day calendar dates according to the above indices
                new_time = [t.replace(d,calendar_365[i]) for t,d,i in zip(time,date,index)]
                # Populate iso_time with new dates
                iso_time = [pd.to_datetime(newt, format='%Y%m-%d%H') for newt in new_time]

            member_data['ISO_TIME'] = iso_time
            member_data = member_data.drop(columns=['YYYY','MM','DD','HH'])

            # Truncate dataset to specified start time
            start = member_data.ISO_TIME.dt.year >= start_year
            member_data = member_data.loc[start]

            track_data = pd.concat([track_data,member_data])
            track_data = track_data.set_index(['ISO_TIME'],append=True)

    # Convert lon/lat from 0/360 to -180/180 if flip_lon is set to True
    if flip_lon:
        track_data['LON'] = (track_data.LON.values + 180) % 360 - 180
        track_data['GEN_LON'] = (track_data.GEN_LON.values + 180) % 360 - 180                                           

    # Create and fit landfrac mesh to lon/lat grid
    lon_landfrac = landfrac.lon.values
    lat_landfrac = landfrac.lat.values
    lon_landfrac_mesh, lat_landfrac_mesh = np.meshgrid(lon_landfrac, lat_landfrac)

    landfrac_points = (lon_landfrac_mesh.flatten(),lat_landfrac_mesh.flatten())
    landfrac_values = landfrac.LANDFRAC.values.flatten()

    track_data = track_data.sort_index()
    print("    Track data loaded!")
    
    return track_data, landfrac_points, landfrac_values



def flatten(l):
    return [item for sublist in l for item in sublist]    


def find_landfalls(storms,hours_over_water=12,states=False,region='GL',threshold=0.5):
    ''' Creates a list of landfalling and non-landfalling storms.
        Parameters: 
            storms (pandas.DataFrame): complete, interpolated storm track data.
            hours_over_water (int): minimum number of hours that a landfalling storm must be over water before another landfall is counted.
            states (bool): when True, extra analysis is performed to categorize landfalls by U.S. state.
            region (str):
            threshold (float):
            
        Returns:
            landfalls (pandas.DataFrame): complete list of landfalling storms.
            landfalls_states (pandas.DataFrame): complete list of U.S. state landfalling storms.
            nonlandfalls (pandas.DataFrame): complete list of non-landfalling storms.
            
    '''
    print("    Generating landfalling/non-landfalling storm lists...")
    
    # Read in US state shapefiles
    # MODIFY FILEPATH ACCORDING TO WHERE YOU STORE YOUR DATA
    if states:
        US_states = shpreader.Reader("/glade/work/abolivar/Pyclogenesis_data/cb_2018_us_state_500k/cb_2018_us_state_500k.shp")
        states_records = US_states.records()
        lst = []

        for state in states_records:
            record = pd.DataFrame.from_dict([state.attributes])
            try:
                geoms = []
                for geom in state.geoemtry.geoms:
                    geoms.append(geom)
                record['geometry'] = [geoms]
            except:
                record['geometry'] = [state.geometry]

            lst.append(record.set_index(['STUSPS','NAME']))

        state_df = pd.concat(lst)
   

    landings_list = []
    nonlandings_list = []
    
    for sid in np.unique(storms.index.get_level_values('SID')): # Iterate through SIDs
        storm_df = storms.xs(sid,level='SID')
        ensemble = storm_df.index.get_level_values('ENUM')[0]
        
        in_region = util.region_bounds(region,storm_df.GEN_LON.values[0],storm_df.GEN_LAT.values[0])
        
        if in_region[0] == True:
            # Check if storm makes landfall
            if (storm_df.LANDFRAC > threshold).any():    
                for itime,time in enumerate(storm_df.index.get_level_values('ISO_TIME').values):
                    if (storm_df.loc[(ensemble,time),'LANDFRAC'] > threshold).all():  # Check if landfrac value is greater than 0.5...
                        # Check if it has been at least X many hours since last landfrac == 1, where X = hours_over water...
                        if itime >= hours_over_water:
                            if (storm_df.iloc[itime-hours_over_water:itime]['LANDFRAC'] <= threshold).all():
                                # Append to landfall list if all of the above conditions are met
                                landings_list.append(storms.loc[[(ensemble,sid,time)]])
                                
                        # Alternative check for storms that make landfall at less than X hours old, where X = hours_over water...
                        else:
                            if (itime > 0) and (storm_df.iloc[0:itime]['LANDFRAC'] <= threshold).all():
                                landings_list.append(storms.loc[[(ensemble,sid,time)]])
                                
            # If the storm did not make landfall, append the lysis point to rhe non-landfall dataframe                   
            else:
                time = storm_df.iloc[-1:].index.get_level_values('ISO_TIME').values[0]
                nonlandings_list.append(storms.loc[[(ensemble,sid,time)]])
                
    landfalls = pd.concat(landings_list,axis=0)
    
    nonlandfalls = pd.concat(nonlandings_list,axis=0)
        
    # Categorize landfalls by state using shapefiles
    if states == True:
        landing_states_list = []
    
        for landing in landings.index:
            landing_copy = landings.loc[landing]
            land_point = shp_geom.Point(landing_copy['LON'],landing_copy['LAT'])

            for state in state_df.index:
                if state_df.loc[state,'geometry'].contains(land_point):
                    landing_state_abbrev = state[0]
                    landing_copy['Location'] = landing_state_abbrev
                    landing_states_list.append(landing_copy)

        landfalls_states = pd.concat(landing_states_list,axis=1).T
        
        return landfalls, landfalls_states, nonlandfalls
    
    print('        Landfalling/non-landfalling storm lists generated!')
    
    return landfalls, nonlandfalls
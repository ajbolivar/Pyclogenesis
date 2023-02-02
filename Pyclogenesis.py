import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.interpolate import griddata
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.io.shapereader as shpreader
import gc
import shapely.geometry as shp_geom
import netCDF4 as nc
import cftime


def tc_list(track_file,data_source,flip_lon=False,calendar='standard',region=None,
            frequency='1H',hours_over_water=12,states=False):
    '''A master function that calls all other functions in Pyclogenesis to return a complete list of storm track data and landfall locations.
       Parameters: 
           track_file (str):
           data_source (str):
           flip_lon (bool):
            
       Returns:
           track_dat (pandas.DataFrame):
           landfrac_points (pandas.DataFrame):
           landfrac_values (pandas.DataFrame):
    '''
    
    track_dat, landfrac_points, landfrac_values = load_track_data(track_file=track_file,data_source=data_source,flip_lon=flip_lon,calendar=calendar,region=region)
    storms = create_storm_list(track_dat=track_dat,landfrac_points=landfrac_points,landfrac_values=landfrac_values,frequency=frequency,calendar=calendar)
    
    if states:
        landfalls, landfalls_states, nonlandfalls = find_landfalls(storms=storms,hours_over_water=hours_over_water,states=states)
        return storms, landfalls, landfalls_states, nonlandfalls
    
    else:
        landfalls, nonlandfalls = find_landfalls(storms=storms,hours_over_water=hours_over_water,states=states)
        return storms, landfalls, nonlandfalls
    
    
    

def load_track_data(track_file,data_source,flip_lon=False,calendar='standard',region=None):
    ''' Using a track data file and a specified data source, formats data to work with Pyclogenesis.
        Parameters: 
            track_file (str): track data file to be processed.
            data_source (str): specification of where data is coming from (options: 'obs','model','reanalysis').
            flip_lon (bool): when True, converts longitude from (0 to 360) degrees to (-90 to 90) degrees (default: False).
            calendar (str): specification of model calendar type (options: 'standard')
            region (str): specification of track region. If None, returns global track dataset (default: None).
            
        Returns:
            track_dat (pandas.DataFrame): raw list of storm tracks
            landfrac_points (pandas.DataFrame): grid of landfrac points
            landfrac_values (pandas.DataFrame): grid of landfrac values
    '''
    # Load land fraction data
    landfrac_dat = '/storage/work/ajb8224/Pyclogenesis_data/landfrac_data/USGS_gtopo30_0.23x0.31_remap_c180612_PHIS_LANDFRAC.nc'
    print("Loading track data located at '{}'...".format(track_file))

    landfrac = xr.open_dataset(landfrac_dat)
    landfrac = landfrac.assign_coords(lon = (((landfrac.lon + 180) % 360) - 180))
    landfrac = landfrac.sortby('lon')
    
    # Observational dataset
    if data_source == 'obs':
        # Read file
        track_dat = pd.read_csv(track_file, usecols = ['SID','ISO_TIME','LAT','LON','WMO_WIND','WMO_PRES'], 
                                index_col=['SID','ISO_TIME'],parse_dates=True)[1:]
        
        track_dat['LAT']   = track_dat['LAT'].astype(float)
        track_dat['LON']   = track_dat['LON'].astype(float)
        track_dat_WMO_WIND = track_dat['WMO_WIND']
        track_dat_WMO_PRES = track_dat['WMO_PRES']
        
        track_dat_WMO_PRES.name = 'PRES'
        track_dat_WMO_WIND.name = 'WIND'
        
        # Replace missing wind and pressure values with NaNs
        track_dat_WMO_WIND_nan = track_dat_WMO_WIND.where(track_dat_WMO_WIND!=' ',np.nan)
        track_dat_WMO_PRES_nan = track_dat_WMO_PRES.where(track_dat_WMO_PRES!=' ',np.nan)
        
        # Remove missing data
        track_dat_WMO_WIND_nan = track_dat_WMO_WIND_nan.dropna() # remove missing data
        track_dat_WMO_PRES_nan = track_dat_WMO_PRES_nan.dropna()
        
        # Make remaining data into floats
        track_dat_WMO_WIND_nan = track_dat_WMO_WIND_nan.astype(float)
        track_dat_WMO_PRES_nan = track_dat_WMO_PRES_nan.astype(float)
        
        track_dat = pd.concat([track_dat[['LAT','LON']],track_dat_WMO_WIND_nan,track_dat_WMO_PRES_nan],axis=1)
        
        # Toss out any times that do not fall on XX:00
        hour_check = pd.to_datetime(track_dat.index.get_level_values(1)).hour.isin(np.arange(0,24,3))
        minute_check = pd.to_datetime(track_dat.index.get_level_values(1)).minute == 0
        second_check = pd.to_datetime(track_dat.index.get_level_values(1)).second == 0

        time_check = (hour_check) & (minute_check) & (second_check)
        track_dat = track_dat.loc[time_check]
        
        print(track_dat)
        
        gen_lat = []
        gen_lon = []
        
        # Generate column of genesis lon/lat information
        for sid in track_dat.index.get_level_values(0):
            storm_df = track_dat.xs(sid,level=0)
            gen_lat.append(storm_df.LAT.values[0])
            gen_lon.append(storm_df.LON.values[0])
        
        # track_dat = track_dat.reset_index()
        track_dat['GEN_LAT'] = gen_lat
        track_dat['GEN_LON'] = gen_lon
        
    # Model/reanalysis datasets
    elif data_source == 'model' or data_source == 'reanalysis':
        # Create unique 3-character strings for all storm IDs
        alphabets = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 
                     'h', 'i', 'j', 'k', 'l', 'm', 'n', 
                     'o', 'p', 'q', 'r', 's', 't', 'u', 
                     'v', 'w', 'x', 'y', 'z']

        indices = []
        for alpha1 in alphabets:
            for alpha2 in alphabets:
                for alpha3 in alphabets:
                    indices.append(alpha1+alpha2+alpha3)
        
        # Read file
        td = open(track_file)   
        track_dat = pd.DataFrame()

        storm_ids = []
        storm_indices = []
        gen_lon = []
        gen_lat = []

        tracks = td.readlines()
        abc = 0
        
        # Parse file into workable data
        for i in range(0, len(tracks)):
            # Use 'start' line to extract information
            if tracks[i].startswith('start'):
                line_items = tracks[i+1].split('\t')
                lon = float(line_items[3])
                lat = float(line_items[4])
                
                # Convert 360 degree longitude to -180/180 degrees
                if flip_lon:
                    lon = (lon + 180) % 360 -180
                
                start_line_items = tracks[i].split('\t')
                
                # If region is defined, check if storm falls within region
                if region != None:
                    in_region = region_bounds(region,lon,lat)
                    if in_region:
                        y,m,d,h = [int(sli) for sli in start_line_items[2:]]
                        sid = '{}{:02d}{:02d}{:02d}_{}'.format(y,m,d,h,indices[abc])
                        
                        for j in range(int(start_line_items[1])):
                            storm_ids.append(sid)
                            gen_lon.append(lon)
                            gen_lat.append(lat)

                        storm_indices.append(i) 
                        abc+=1

                    else:
                        storm_data_length = int(start_line_items[1])
                        skip_array = np.arange(i, i+2+storm_data_length, 1)

                        for n in skip_array:
                            storm_indices.append(n)
                            
                # If region is not defined, append all storms           
                else:
                    y,m,d,h = [int(sli) for sli in start_line_items[2:]]
                    sid = '{}{:02d}{:02d}{:02d}_{}'.format(y,m,d,h,indices[abc])

                    for j in range(int(start_line_items[1])):
                        storm_ids.append(sid)
                        gen_lon.append(lon)
                        gen_lat.append(lat)

                    storm_indices.append(i)
                    abc+=1
 
        # Create DataFrame from track file, skip header rows (those beginning with start)
        track_dat = pd.read_table(track_file, skiprows=storm_indices, usecols=np.arange(3,12,1),
                                       delimiter = '\t', header=None, 
                                       names=['LON','LAT','PRES','WIND','LANDFRAC','YYYY','MM','DD','HH'])
        
        track_dat['SID'] = storm_ids
        track_dat['GEN_LON'] = gen_lon
        track_dat['GEN_LAT'] = gen_lat
        track_dat['PRES'] = (track_dat['PRES'].values)/100
        track_dat = track_dat.set_index(['SID']).sort_index()
        
        iso_time = []
        
        # Format times into a pandas datetime
        for sid in np.unique(track_dat.index.get_level_values(0))[::]:
            storm = track_dat.xs(sid).reset_index()

            if isinstance(storm,pd.Series):
                storm = storm.to_frame().T

            for time in storm.index:
                y, m, d, h = [int(storm.loc[time,'YYYY']), int(storm.loc[time,'MM']), 
                              int(storm.loc[time,'DD']), int(storm.loc[time,'HH'])]
                t = '{}{:02d}{:02d}{:02d}'.format(y,m,d,h)

                if calendar == 'standard':
                    iso_time.append(pd.to_datetime(t, format='%Y%m%d%H'))

                else:
                    iso_time.append(int(t))

        track_dat['ISO_TIME'] = iso_time
        track_dat = track_dat.drop(columns=['YYYY','MM','DD','HH'])
        
        # Set additional indices: ISO_TIME, GEN_LON, GEN_LAT
        track_dat = track_dat.set_index(['ISO_TIME','GEN_LON','GEN_LAT'],append=True)
    
    # Convert lon/lat from 0/360 to -180/180 if flip_lon is set to True
    if flip_lon:
        lon_landfrac = (landfrac.lon.values + 180) % 360 - 180
        track_dat['LON'] = (track_dat.LON.values + 180) % 360 - 180
    else:
        lon_landfrac = landfrac.lon.values
        
    # Create and fit landfrac mesh to lon/lat grid
    lat_landfrac = landfrac.lat.values
    lon_landfrac_mesh, lat_landfrac_mesh = np.meshgrid(lon_landfrac, lat_landfrac)

    landfrac_points = (lon_landfrac_mesh.flatten(),lat_landfrac_mesh.flatten())
    landfrac_values = landfrac.LANDFRAC.values.flatten()
    
    track_dat = track_dat.sort_index()
    print("Track data loaded!")
    
    return track_dat, landfrac_points, landfrac_values

def region_bounds(region,lon,lat):
    ''' Determines whether a storm genesis point is within a specified geographic region.
        Parameters: 
            region (str):
            lon (str):
            lat (str):
            
        Returns:
            in_region (bool):
    '''
    in_region = False
    
    # North Atlantic
    if region == 'NA':
        if (-100 < lon) and (lon < -1) and (0 < lat) and (lat < 50):
            if ((lon < -85) and (lat < 16)) or ((lon < -91) and (lat < 18)):
                in_region = False
            else:
                in_region = True

    ### INCOMPLETE
    # East Pacific
    if region == 'EP':
        in_region = False
        
    # West Pacific    
    if region == 'WP':
        in_region = False
        
    # Indian Ocean    
    if region == 'IO':
        in_region = False
        
    # Oceania
    if region == 'OC':
        in_region = False
        
    return in_region

def interp_location_landfrac(storm_df,landfrac_points,landfrac_values,frequency,calendar='standard'):
    ''' Interpolates track dataset to a specified time frequency.
        Parameters: 
            storm_df (pandas.DataFrame): raw storm track data.
            landfrac_points (pandas.DataFrame): grid of land fraction points.
            landfrac_values (pandas.DataFrame): grid of land fraction values.
            frequency (str): time frequency for interpolation.
            calendar (str): specification of model calendar type (options: 'standard').
            
        Returns:
            interped_df (pandas.DataFrame): interpolated storm track data.
    '''
    
    if calendar == 'standard':
        storm_times = storm_df.index.get_level_values('ISO_TIME')
        # Set desired time range (according to desired frequency)
        storm_time_hourly = pd.date_range(storm_times[0],storm_times[-1],freq=frequency)
        temp_df = storm_df.copy()
        temp_df = temp_df.reset_index()
        
        # Create new ISO_TIME column the length of storm_time_hourly, input existing times at appropriate indices
        for time in storm_time_hourly:
            length = len(temp_df)
            if time not in storm_times:
                temp_df.loc[length,'ISO_TIME'] = time
        
        temp_df = temp_df.set_index(['ISO_TIME'])
        temp_df.index = pd.to_datetime(temp_df.index)
        temp_df = temp_df.sort_index()
        
        # Linearly interpolate DataFrame
        interped_df   = temp_df.interpolate(axis='index',method='linear')
        temp_points   = list(zip(interped_df['LON'].values,interped_df['LAT'].values)) # Put points together as a list 
        temp_landfrac = griddata(landfrac_points, landfrac_values, temp_points, # interpolate landfrac
                                 method = 'nearest')

        interped_df['LANDFRAC'] = temp_landfrac
            
    ### INCOMPLETE ###
    else:
        temp_df = pd.DataFrame()
        interp = []
        
        for sid in np.unique(storm_df.index.get_level_values(0))[::]:
            storm = storm_df.xs(sid)
            length = storm.shape(1)
            
            storm_array = storm.to_numpy()
    
    return interped_df
    
    
def create_storm_list(track_dat,landfrac_points,landfrac_values,frequency='1H',
                      num_storms=None,calendar='standard'):
    ''' Creates complete list of storm tracks with all associated information.
        Parameters:
            track_dat (pandas.DataFrame):
            landfrac_points (pandas.DataFrame):
            landfrac_values (pandas.DataFrame):
            frequency (str):
            num_storms (int):
            calendar (str):
            
        Returns:
            storms (pandas.DataFrame):
            
    '''
    print("Generating storm list...")
    
    storm_sid_land_list = []                                # Array of pandas dataframes
    count = 0

    for sid in np.unique(track_dat.index.get_level_values(0))[::]:   # Loop over SID
        storm_sid        = track_dat.xs(sid,level='SID').copy()  # Select the storm whose id is sid
        storm_sid_interp = interp_location_landfrac(storm_sid,landfrac_points,landfrac_values,frequency)
        storm_lats       = storm_sid_interp['LAT'].values        # Retrieve lats and lons for individual storm
        storm_lons       = storm_sid_interp['LON'].values
        storm_landfrac   = storm_sid_interp['LANDFRAC'].values
        storm_wind       = storm_sid_interp['WIND'].values
        storm_pres       = storm_sid_interp['PRES'].values

        storm_sid_interp['LANDFRAC']       = storm_landfrac      # Add interpolated landfrac to dataframe
        storm_sid_interp['SID']            = sid

        storm_sid_interp = storm_sid_interp.set_index('SID','ISO_TIME', append=True)
        storm_sid_interp = storm_sid_interp.reorder_levels(['SID','ISO_TIME'])

        storm_sid_land_list.append(storm_sid_interp)
        
        if num_storms is not None:
            if count > num_storms:
                break
            
            count += 1
        

    storms = pd.concat(storm_sid_land_list,axis=0) # Combine all storms into one dataframe again
    
    storms.loc[storms['LANDFRAC']<=.5,'LANDFRAC'] = 0
    storms.loc[storms['LANDFRAC']>.5,'LANDFRAC'] = 1
    
    print('Storm list generated!')
    
    return storms


                        
def find_landfalls(storms,hours_over_water=12,states=False):
    ''' Creates a list of landfalling and non-landfalling storms.
        Parameters: 
            storms (pandas.DataFrame):
            hours_over_water (int):
            states (bool):
            state_df (pandas.DataFrame):
            
        Returns:
            landfalls (pandas.DataFrame):
            landfalls_satellite (pandas.DataFrame):
            landfalls_states (pandas.DataFrame):
            nonlandfalls (pandas.DataFrame):
            
    '''
    print("Generating landfalling/non-landfalling storm list...")
    
    # Read in US state shapefiles
    if states:
        US_states = shpreader.Reader("/storage/work/ajb8224/Pyclogenesis_data/cb_2018_us_state_500k/cb_2018_us_state_500k.shp")
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
    
    for sid in storms.index.levels[0]: # Iterate through SIDs
        storm_df = storms.xs(sid,level=0)
        
        for itime, time in enumerate(storm_df.index.values): # For all times and time indices in storm_df...
            if (storm_df.loc[time,'LANDFRAC'] > 0.5).all():  # Check if landfrac value is greater than 0.5...
                # Check if it has been at least X many hours since last landfrac == 1, where X = hours_over water...
                if itime >= hours_over_water:
                    # Append to landfall list if all of the above conditions are met
                    if (storm_df.iloc[itime-hours_over_water:itime]['LANDFRAC'] <= 0.5).all():
                        landings_list.append(storms.loc[[(sid,time)]])
                            
                # Alternative check for storms that make landfall at less than X hours old, where X = hours_over water...
                else:
                    if (itime > 0) and (storm_df.iloc[0:itime]['LANDFRAC'] <= 0.5).all():
                        landings_list.append(storms.loc[[(sid,time)]])  
                            
            elif itime == (len(storm_df)-1):
                nonlandings_list.append(storms.loc[[(sid,time)]])
                    
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
    
    print('Landfalling/non-landfalling storm lists generated!')
    
    return landfalls, nonlandfalls


                        
def storm_track_plt(ax,storms,storm_numbers,landings,size=1.2,plot_type='wind_speed',
                    scattercolor='blue',linecolor='gainsboro',linewidth=0.25,legend=False,
                    legend_colors=None,track_labels=None):
    ''' Plots storm tracks, wind speed, and landfall points (if applicable) by location.
        Parameters: 
            
        Returns:
            
    '''
    storm_list = np.unique(storms.index.get_level_values(0))
    ws=False
    
    for storm in storm_numbers:
        storm_number_id = storm_list[storm]
        storm_number_data = storms.xs(storm_number_id,level=0)

        x = storm_number_data.LON.values
        y = storm_number_data.LAT.values
        
        # Plot wind speed intensity along track
        if plot_type == 'wind_speed':
            ws=True
            # Set categories for storm intensity by wind speed
            low_winds  = [0, 34,64,83, 96,113,137]
            high_winds = [33,63,82,95,112,136,np.inf]
            colors = ['dodgerblue','mediumturquoise','yellow','gold','darkorange','red','magenta']
            landfrac_colors = ['purple','yellow']
            labels = ['TD','TS','Cat 1','Cat 2','Cat 3','Cat 4','Cat 5']
            
            for low_wind,high_wind,color in zip(low_winds,high_winds,colors):
                storm_winds = storm_number_data[storm_number_data.WIND.between(low_wind,high_wind,inclusive='both')]
                x_wind = storm_winds.LON.values
                y_wind = storm_winds.LAT.values

                # Plot wind speeds along track
                ax.plot(x,y,color='gainsboro',linewidth=linewidth,transform=ccrs.PlateCarree())
                ax.scatter(x_wind,y_wind,s=size,c=color,transform=ccrs.PlateCarree())
                
            if legend:   
                handles = []
                for color in colors:
                    point = Line2D(range(1),range(1),color='white',marker='o',markerfacecolor=color,
                                   markeredgecolor='none',markersize=0.75,linewidth=0)
                    handles.append(point)

                labels.append('Landfall')
                handles.append(Line2D(range(1),range(1),color='white',marker='o',markerfacecolor='white',
                                      markeredgecolor='white',markersize=0.75,linewidth=0))

                ax.legend(handles=handles,ncol=1,fontsize='xx-small',loc='upper right',markerscale=4,labels=labels)
                
        # Plot landfrac values along track    
        elif plot_type == 'landfrac':
            ax.plot(x,y,color='gainsboro',linewidth=linewidth,transform=ccrs.PlateCarree())
            ax.scatter(x_wind,y_wind,s=size,c=linecolor,transform=ccrs.PlateCarree())
           
        # Plot solid color track
        elif plot_type == 'solid':
            ax.plot(x,y,color=linecolor,linewidth=linewidth,transform=ccrs.PlateCarree())
            ax.scatter(x,y,s=size,c=scattercolor,transform=ccrs.PlateCarree(),marker='.')
            
            if legend:
                handles = []
                for color in color_list:
                    point = Line2D(range(1),range(1),color='white',marker='o',markerfacecolor=color,
                                   markeredgecolor='none',markersize=0.75,linewidth=0)
                    handles.append(point)

                track_labels.append('Landfall')
                handles.append(Line2D(range(1),range(1),color='white',marker='o',markerfacecolor='white',
                                      markeredgecolor='white',markersize=0.75,linewidth=0))
                
                ax.legend(handles=handles,ncol=1,fontsize='xx-small',loc='upper right',markerscale=4,labels=track_labels) 
               
        # Plot landfall locations
        if storm_number_id in landings.index.get_level_values(0):
            storm_landings = landings.xs(storm_number_id)
            ax.scatter(storm_landings.LON,storm_landings.LAT,s=size*2,facecolor='none',edgecolor='white',zorder=5)

            if len(storms) < 5:
                print("Storm {} made landfall at the following locations:".format(storm_number_id))
                print(storm_landings)
                
        else:
            if len(storms) < 5:
                print("Storm {} did not make landfall.".format(storm_number_id))
                
    # Return SID if only plotting one storm
    if len(storm_numbers) == 1:
        return storm_number_id

                        

def mappy(ax,extent=[-100,-10,10,60],lat_ticks=15,lon_ticks=30,crs=ccrs.PlateCarree(),fontsize='x-small',title=None,
          borders='black',states_color='#2b5714',state_borders='black',ocean_color='#062045',land_color='#2b5714'):
    ''' Decorates plot to look more "map-like".
        Parameters: 
            ax:
            extent (int arr):
            lat_ticks (float):
            lon_ticks (float):
            crs (cartopy projection):
            
        Returns:
            None
    '''
    # Add extent    
    lonW, lonE, latS, latN = extent[0], extent[1], extent[2], extent[3]

    # Add lat/lon ticks
    ax.set_yticks(np.arange(latS+10,latN,lat_ticks), crs=ccrs.PlateCarree(), minor=False)
    ax.set_yticks(np.arange(latS+10,latN,lat_ticks/2), crs=ccrs.PlateCarree(), minor=True)
    
    ax.set_xticks(np.arange(lonW,lonE,lon_ticks), crs=ccrs.PlateCarree(), minor=False)
    ax.set_xticks(np.arange(lonW,lonE,lon_ticks/2), crs=ccrs.PlateCarree(), minor=True)
    
    ax.set_xticklabels(np.arange(lonW,lonE,lon_ticks),fontsize=fontsize)
    ax.set_yticklabels(np.arange(latS+10,latN,lat_ticks),fontsize=fontsize)
    
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    
    ax.set_extent(extent,crs=ccrs.PlateCarree())
    
    if title != None:
        ax.title(title)
    
    ax.add_feature(cfeature.LAND.with_scale('10m'),facecolor=land_color,edgecolor=borders,linewidth=0.25)
    ax.add_feature(cfeature.OCEAN.with_scale('10m'),facecolor=ocean_color,edgecolor=borders,linewidth=0.25)
    ax.add_feature(cfeature.BORDERS,facecolor='none',edgecolor=borders,linewidth=.25)
    ax.add_feature(cfeature.COASTLINE,facecolor='none',edgecolor=borders,linewidth=.25)
    ax.add_feature(cfeature.STATES,facecolor=states_color,edgecolor=state_borders,linewidth=.25)
    ax.add_feature(cfeature.LAKES.with_scale('110m'),edgecolor=borders,facecolor=ocean_color,linewidth=.25)

                        

def manual_regional_lf_plt(ax,landfalls,extent=[-100,-60,15,50],color='white',size=10,lw=0.25,zorder=100):
    ''' Plots landfalls within an area specified by a user-defined lat/lon box.
        Parameters: 
            ax:
            landfalls (pandas.DataFrame):
            extent (int arr):
            color (str):
            size (float):
            lw (float):
            zorder (int):
        Returns:
            None
    '''
    
    # Define region bounds
    lonW, lonE, latS, latN = extent[0], extent[1], extent[2], extent[3]

    isel_lat = np.logical_and(landfalls.LAT < latN, landfalls.LAT >= latS)
    isel_lon = np.logical_and(landfalls.LON < lonE, landfalls.LON >= lonW)
    isel_tot = np.logical_and(isel_lat, isel_lon)

    landings_sub = landfalls.loc[isel_tot]
    
    ax.scatter(landings_sub.LON,landings_sub.LAT,facecolor=color,edgecolor='black',s=size,
               linewidth=lw,zorder=zorder)

    
    
def track_density_plot(storms,bins=(np.arange(-120,0,4), np.arange(0,70,4)),cmap='inferno'):
    ''' Plots a 2D histogram displaying track density
    Parameters: 
        ax (arr):
        storms (pandas.DataFrame):
        bins (int or arr)):
        cmap (str):
    Returns:
        plot:
    '''
    h,x,y = np.histogram2d(storms.LON.values,storms.LAT.values, bins = bins)
    #h,x,y,plot = ax.hist2d(storms.LON.values,storms.LAT.values, bins=bins, cmap=cmap)
    #return h
    return h, x, y
    

    
def gen_to_lf_plot(ax,landfalls,nonlandfalls=None,
                  gen_south=10,gen_north=25,gen_west=-80,gen_east=-20,size=2.5,
                  gen_color_lf='blue',landfall_color='limegreen',gen_color_nlf='yellow',nonlandfall_color='red',
                  edgecolor='white',linewidth=0.5):
    ''' Plots scatter points based on a user-defined storm genesis region.
    Parameters: 
        ax (arr):
        landfalls (pandas.DataFrame):
        nonlandfalls (pandas.DataFrame)
        gen_south (float):
        gen_north (float):
        gen_west (float):
        gen_east (float):
        size (float):
        gen_color_lf (str):
        landfall_color (str):
        gen_color_nlf (str):
        nonlandfall_color (str)
        edgecolor(str):
        linewidth (float):
    Returns:
        None
    '''
    try:
        # Iterate through indiviadual storms in landfalls DataFrame
        for sid in landfalls.index.levels[0]:
            storm_df = landfalls.xs(sid,level=0)
            GEN_LAT  = storm_df.GEN_LAT.values[0]
            GEN_LON  = storm_df.GEN_LON.values[0]

            # MDR: 10-20N, 80-20W
            gen_S  = gen_south
            gen_N  = gen_north
            gen_W  = gen_west
            gen_E  = gen_east

            lat_check = (GEN_LAT>gen_S) & (GEN_LAT<gen_N)
            lon_check = (GEN_LON>gen_W) & (GEN_LON<gen_E)
            
            # Check if storm genesis point falls within defined region
            if lat_check & lon_check:
                # Plot genesis point
                ax.scatter(GEN_LON,GEN_LAT,s=size,facecolor=gen_color,edgecolor=edgecolor,
                           zorder=5,linewidths=linewidth)
                # Plot landfall point
                ax.scatter(storm_df.LON.values,storm_df.LAT.values,s=size,facecolor=landfall_color,
                           edgecolor=edgecolor,zorder=5,linewidths=linewidth)
                
        # Iterate through indiviadual storms in nonlandfalls DataFrame
        if nonlandfalls != None:
            for sid in nonlandfalls.index.levels[0]:
                storm_df = nonlandfalls.xs(sid,level=0)
                GEN_LAT  = storm_df.GEN_LAT.values[0]
                GEN_LON  = storm_df.GEN_LON.values[0]

                # MDR: 10-20N, 80-20W
                gen_S  = gen_south
                gen_N  = gen_north
                gen_W  = gen_west
                gen_E  = gen_east

                lat_check = (GEN_LAT>gen_S) & (GEN_LAT<gen_N)
                lon_check = (GEN_LON>gen_W) & (GEN_LON<gen_E)

                # Check if storm genesis point falls within defined region
                if lat_check & lon_check:
                    # Plot genesis point
                    ax.scatter(GEN_LON,GEN_LAT,s=size,facecolor=gen_color,edgecolor=edgecolor,
                               zorder=5,linewidths=linewidth)
                    # Plot nonlandfall point
                    ax.scatter(storm_df.LON.values,storm_df.LAT.values,s=size,facecolor=nonlandfall_color,
                               edgecolor=edgecolor,zorder=5,linewidths=linewidth)
                
    # If no points within given bounds are detected
    except:
        print('Model has no storm genesis points in the defined region.')
          
            
            
def lf_to_gen_plot(ax,landfalls=None,nonlandfalls=None,lf=True,nlf=True,
                  lf_south=10,lf_north=25,lf_west=-80,lf_east=-20,size=2.5,
                  gen_color_lf='blue',lf_color='limegreen',gen_color_nlf='yellow',nlf_color='red',
                  lf_edge='black',nlf_edge='black',gen_edge = 'black',linewidth=0.5):
    ''' Plots scatter points based on a user-defined storm landfall region.
    Parameters: 
        ax (arr):
        landfalls (pandas.DataFrame):
        nonlandfalls (pandas.DataFrame)
        gen_south (float): southern bound of region box (range from -90 to 90)
        gen_north (float): northern bound of region box (range from -90 to 90)
        gen_west (float): western bound of region box (range from -180 to 180)
        gen_east (float): eastern bound of region box (range from -180 to 180)
        size (float): size of scatter point
        gen_color_lf (str): color of genesis point of landfalling storm
        landfall_color (str): color of landfall point
        gen_color_nlf (str): color of genesis point of storm that does not make landfall
        nonlandfall_color (str): color of cyclolysis point over water
        edgecolor(str): color of scatter point edge
        linewidth (float): width of scatter point edge
    Returns:
        None
    '''
    
    try:
        if lf:
            # print('pass')
            for sid in landfalls.index.get_level_values(0):
                # print('pass1')
                storm_df = landfalls.xs(sid,level=0)
                LF_LAT  = storm_df.LAT.values[0]
                LF_LON  = storm_df.LON.values[0]
                # print('pass2')
                # MDR: 10-20N, 80-20W
                lf_S  = lf_south
                lf_N  = lf_north
                lf_W  = lf_west
                lf_E  = lf_east

                lat_check = (LF_LAT>lf_S) & (LF_LAT<lf_N)
                lon_check = (LF_LON>lf_W) & (LF_LON<lf_E)
                # print('pass3')
                if lat_check & lon_check:
                    # print('pass4')
                    ax.scatter(LF_LON,LF_LAT,s=size,facecolor=lf_color,edgecolor=lf_edge,
                               zorder=5,linewidths=linewidth)
                    ax.scatter(storm_df.GEN_LON.values,storm_df.GEN_LAT.values,s=size,facecolor=gen_color_lf,
                               edgecolor=gen_edge,zorder=5,linewidths=linewidth)
        
        if nlf:
            for sid in nonlandfalls.index.get_level_values(0):
                # print('pass1')
                storm_df = nonlandfalls.xs(sid,level=0)
                LF_LAT  = storm_df.LAT.values[0]
                LF_LON  = storm_df.LON.values[0]
                # print('pass2')
                # MDR: 10-20N, 80-20W
                lf_S  = lf_south
                lf_N  = lf_north
                lf_W  = lf_west
                lf_E  = lf_east

                lat_check = (LF_LAT>lf_S) & (LF_LAT<lf_N)
                lon_check = (LF_LON>lf_W) & (LF_LON<lf_E)
                # print('pass3')
                if lat_check & lon_check:
                    # print('pass4')
                    ax.scatter(LF_LON,LF_LAT,s=size,facecolor=nlf_color,edgecolor=nlf_edge,
                               zorder=5,linewidths=linewidth)
                    ax.scatter(storm_df.GEN_LON.values,storm_df.GEN_LAT.values,s=size,facecolor=gen_color_nlf,
                               edgecolor=gen_edge,zorder=5,linewidths=linewidth)
    
    # If no points within given bounds are detected
    except:
        print('Model has no storm landfall points in the defined region.')

    
def regional_lf_plot(ax,landfalls,landing_states,state_names,colors=['white'],size=4,lw=0.25,zorder=100):
    ''' Plots landfalls within a user-specified U.S. state/territory.
        Parameters: 
            ax:
            landfalls (pandas.DataFrame):
            landing_states (pandas.DataFrame):
            state_name (str):
            color (str):
            size (float):
            lw (float):
            zorder (int):
            
        Returns:
            None
    '''
    for state, color in zip(state_names,colors):
        lf = landing_states[landing_states['Location']==state]
        ax.scatter(lf.LON,lf.LAT,facecolor=color,edgecolor='black',s=size,linewidth=lw,zorder=zorder)

                    

def temporal_lf_plot(ax,landfalls,size=5,lw=0.25, pre_col='white',early_col='yellow',peak_col='red',late_col='blue',post_col='white',edgecolor='black'):
    ''' Plots landfalls color-coded by seasonal timing (pre-season, early season, peak season,
        late season, post season).
    Parameters: 
        ax:
        landfalls (pandas.DataFrame):
        size (float):
        lw (float):
        pre_col (str):
        early_col (str):
        peak_col (str):
        post_col (str):
        edgecolor (str):
    Returns:
        None
    '''
    
    # Pre-season: March, April, May
    pre_szn = landfalls[pd.to_datetime
                                 (landfalls.index.get_level_values(1)).month.isin([3,4,5])]
    
    # Early season: June, July
    early_szn = landfalls[pd.to_datetime
                                   (landfalls.index.get_level_values(1)).month.isin([6,7])]
    
    # Peak season: August, September
    peak_szn = landfalls[pd.to_datetime
                                  (landfalls.index.get_level_values(1)).month.isin([8,9])]
    
    # Late season: October, November
    late_szn = landfalls[pd.to_datetime
                                  (landfalls.index.get_level_values(1)).month.isin([10,11])]
    
    # Post-season: December, January, February
    post_szn = landfalls[pd.to_datetime
                                  (landfalls.index.get_level_values(1)).month.isin([12,1,2])]
    
    categories = [pre_szn, early_szn, peak_szn, late_szn, post_szn]
    labels = ['MAM','JJ','AS','ON','DJF']
    
    ax.scatter(pre_szn.LON, pre_szn.LAT, c=pre_col, edgecolor=edgecolor, linewidth=lw, s=size, zorder=2, label=labels[0])
    ax.scatter(early_szn.LON, early_szn.LAT, c=early_col, edgecolor=edgecolor, linewidth=lw, s=size, zorder=3, label=labels[1])
    ax.scatter(peak_szn.LON, peak_szn.LAT, c=peak_col, edgecolor=edgecolor, linewidth=lw, s=size, zorder=4, label=labels[2])
    ax.scatter(late_szn.LON, late_szn.LAT, c=late_col, edgecolor=edgecolor, linewidth=lw, s=size, zorder=3, label=labels[3])
    ax.scatter(post_szn.LON, post_szn.LAT, c=post_col, edgecolor=edgecolor, linewidth=lw, s=size, zorder=2, label=labels[4])
    
    ax.legend(ncol=1,fontsize='xx-small',loc='upper right',markerscale=2)
    


def storm_statistics(storms,landfalls,name,end_year=None,region=None,convert_pres=False):
    ''' Creates a csv containing several climatological TC statistics: avg TC count per year, avg landfall count per year, avg number of TC days per year, avg storm lifetime before landfall, med storm lifetime before landfall
        Parameters: 
        storms (pandas.DataFrame):
        landfalls (pandas.DataFrame:
        name (string):
        end_year (int):
    Returns:
        None
    '''
    # Length of dataset (in years)
    dataset_yrs = np.unique(storms.index.get_level_values(1).astype('str').str[:4].astype('int').values)
    # print(dataset_yrs)
    
    # Truncate dataset to desired length
    if end_year != None:
        length = len(np.where(np.logical_and(dataset_yrs >= 1979, dataset_yrs <= end_year))[0])
    else:
        length = len(np.where(dataset_yrs >= 1979)[0])
    
    
    # Average annual TC count
    yearly_count = np.round((len(np.unique(storms.loc[storms.index.get_level_values(0).str[:4].astype('int').values>=1979].index.get_level_values(0))))/length,decimals=1)
    
    # Average annual landfall count
    yearly_landfall_count = np.round((len(np.unique(landfalls.loc[landfalls.index.get_level_values(0).str[:4].astype('int').values>=1979].index.get_level_values(0))))/length,decimals=1)
    
    # Average annual number of TC days
    dates = []
    for date in (np.unique(storms.index.get_level_values(1).astype('str').str[:10])) :
        dates.append(int(date[:4]))

    dates = np.asarray(dates)
    yearly_days = np.round((len(np.where(dates >= 1979)[0])/length),decimals=1)

    # Average storm lifetime before landfall (in days)
    lifetimes = []
    for sid in (np.unique(landfalls.index.get_level_values(0))):
        # print(sid)
        storm_df = storms.xs(sid,level=0)
        datelist = []
        dates = storm_df.index.get_level_values(0).astype('str').str[:10]

        for date in dates:
            if date not in datelist:
                datelist.append(date)
        
        lifetimes.append(len(datelist))
    
    med_lifetime = np.median(lifetimes)
    avg_lifetime = np.round(np.mean(lifetimes),decimals=1)

    print('======={}======='.format(name))
    print('Average annual TC count: {}\nAverage annual landfall count: {}\nAverage annual TC days: {}\nAverage storm lifetime before landfall (days): {}\nMedian storm lifetime before landfall (days): {}'
          .format(yearly_count,yearly_landfall_count,yearly_days,avg_lifetime,med_lifetime))
    
    stats = [name,yearly_count,yearly_landfall_count,yearly_days,avg_lifetime,med_lifetime]
    stats = pd.DataFrame([stats],columns=['NAME','AVG_COUNT','AVG_LF','AVG_DAYS','AVG_LIFE','MED_LIFE'])
    
    stats.to_csv('{}_TC_stats.csv'.format(name),index=False)
    return stats
    
    



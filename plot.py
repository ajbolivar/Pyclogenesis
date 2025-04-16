import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.colors as colors
from collections import Counter
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from Pyclogenesis import util

# Categorize landfall by region
def categorize_landfall(landfall, plot_type):
    # Landfall regions
    gc = Polygon([(-98, 17), (-98, 33), (-82, 33), (-81, 25), 
                  (-83, 25), (-83, 24), (-90, 24), (-90, 17)]) # Gulf Coast
    se = Polygon([(-83, 24), (-83, 25), (-81, 25), (-82, 33.5), 
                  (-80.5, 36.5), (-74.5, 36.5), (-74.5, 27), 
                  (-79.5, 27), (-79.5, 25), (-80, 24)])
    ne = Polygon([(-78, 36.5), (-78, 48), (-59, 48), (-59, 36.5)])
    ci_ca = Polygon([(-90., 15), (-90., 24), (-79.5, 24), (-79.5, 27), 
                     (-74.5, 27), (-58.5, 19.5), (-58.5, 11), (-85, 11)])
    # Genesis regions
    mdr = Polygon([(-60,10), (-60,20), (-15,20), (-15,10)])
    ws = Polygon([(-81.5,25), (-81.5, 45), (-43,45), (-43,20), (-60,20)])
    gm = Polygon([(-98,17), (-98,31), (-81.5,31), (-81.5,25), (-93,17)])
    cs = Polygon([(-93,17), (-81.5,25), (-60,20), (-60,10), (-83,10)])
    # NATL basin
    natl = Polygon([(-83,10),(-93,17),(-98,17),(-98,45),(-15,45),(-15,10)])
    
    if plot_type == 'gen_to_lf':
        genesis_point = Point(landfall.gen_lon, landfall.gen_lat)
        if mdr.contains(genesis_point): return 'mediumvioletred', 'deeppink', 'mdr'
        elif ws.contains(genesis_point): return 'darkblue', 'dodgerblue', 'ws'
        elif gm.contains(genesis_point): return 'darkgreen', 'limegreen', 'gm'
        elif cs.contains(genesis_point): return 'darkgoldenrod', 'gold', 'cs'
        elif natl.contains(genesis_point): return 'darkgray', 'gainsboro', 'other_natl'
        else: 
            print(genesis_point)
            return 'none', 'none', 'outside'
             
    if plot_type == 'lf_to_gen':
        landfall_point = Point(landfall.lon, landfall.lat)
        if gc.contains(landfall_point): return 'darkred', 'red', 'gc'
        elif se.contains(landfall_point): return 'mediumblue', 'deepskyblue', 'se'
        elif ne.contains(landfall_point): return 'rebeccapurple', 'darkviolet', 'ne'
        elif ci_ca.contains(landfall_point): return 'saddlebrown', 'darkorange', 'ci_ca'
        elif natl.contains(landfall_point): return 'darkgray', 'gainsboro', 'other_natl'
        else: return 'none','none', 'outside'

# Create table with statistics associated with genesis and landfall to accompany spaghetti plot
def spaghetti_table(landfalls=None, nonlandfalls=None, region='NA', plot_type='gen_to_lf', lf=True, nlf=True):
    subregions = []
    if lf:
        landfalls = util.subset_storms(region, landfalls)
        for enum in np.unique(landfalls.index.get_level_values('enum')):
            ens_lfs = (landfalls.xs(enum, level='enum', drop_level=False)).reset_index()    
            for index, lf in ens_lfs.iterrows():
                _, _, subregion = categorize_landfall(lf, plot_type=plot_type)
                subregions.append(subregion)
    if nlf:
        for enum in np.unique(nonlandfalls.index.get_level_values('enum')):
            ens_nlfs = (nonlandfalls.xs(enum, level='enum', drop_level=False)).reset_index()
            for index, nlf in ens_nlfs.iterrows():
                _, _, subregion = categorize_landfall(nlf, plot_type=plot_type)
                subregions.append(subregion)


    subregions_count = Counter(subregions)
    return subregions_count

# Plots storm tracks based on a user-defined storm landfall region
def gen_to_lf(ax, storms, landfalls=None, nonlandfalls=None, lf=True, nlf=True, s=2.5, lw=0.5, a=1, div=4):
    if lf:
        lf_inds = range(len(landfalls))
        # Find average number of landfalls per ensemble member
        try:
            avg_num_lf = int(np.round((len(lf_inds)/len(np.unique(landfalls.index.get_level_values('enum'))))/div, 0))
        except:
            avg_num_lf = len(lf_inds)

        # Randomly subset landfalls to plot based on avg_num_lf   
        random_subset = np.random.choice(lf_inds, avg_num_lf, replace=False)
        temp_df = []
            
        for i in random_subset:
            # Extract landfall/storm data from parent dataframes
            temp_df.append(landfalls.iloc[[i]])
        
        landfalls = pd.concat(temp_df)
        
        for enum in np.unique(landfalls.index.get_level_values('enum')):
            ens_lfs = (landfalls.xs(enum,level='enum', drop_level=False)).reset_index()
            ens_stms = (storms.xs(enum,level='enum', drop_level=False)).reset_index()
            
            for index, lf in ens_lfs.iterrows():
                sid = lf.sid
                storm_df = ens_stms[ens_stms.sid == sid]
                
                # Make landfall data into a tuple
                lf_tuple = tuple(lf)
                
                # Find index in storm_df corresponding to landfall
                lf_index = (storm_df[storm_df.apply(tuple, axis=1) == lf_tuple]).index.to_list()[0]

                # Crop the storm_df dataframe using this index
                cropped_storm_df = storm_df.loc[:lf_index]
                facecolor, edgecolor, _ = categorize_landfall(lf, plot_type='gen_to_lf')

                # Plot genesis point
                ax.scatter(lf.lon, lf.lat, s=s, facecolor=facecolor,
                           edgecolor=edgecolor, zorder=5, linewidths=lw, alpha=a)
                # Plot storm track
                ax.plot(cropped_storm_df.lon.values, cropped_storm_df.lat.values, lw=lw*2, color=edgecolor, zorder=-1)

            
        
    if nlf:
        nlf_inds = range(len(nonlandfalls))
        # Find average number of nonlandfalls per ensemble member
        try:
            avg_num_nlf = int(np.round((len(nlf_inds)/len(np.unique(nonlandfalls.index.get_level_values('enum'))))/div, 0))
        except:
            avg_num_nlf = len(lf_inds)

        # Randomly subset nonlandfalls to plot based on avg_num_lf   
        random_subset = np.random.choice(nlf_inds, avg_num_nlf, replace=False)
        temp_df = []
            
        for i in random_subset:
            # Extract nonlandfall data from parent dataframe
            temp_df.append(nonlandfalls.iloc[[i]])
                
        nonlandfalls = pd.concat(temp_df)
        for enum in np.unique(nonlandfalls.index.get_level_values('enum')):
            ens_nlfs = (nonlandfalls.xs(enum, level='enum', drop_level=False)).reset_index()
            ens_stms = (storms.xs(enum, level='enum', drop_level=False)).reset_index()
                
            for index, nlf in ens_nlfs.iterrows():
                sid = nlf.sid
                storm_df = ens_stms[ens_stms.sid == sid]
                
                # Make landfall data into a tuple
                nlf_tuple = tuple(nlf)
                
                # Find index in storm_df corresponding to landfall
                nlf_index = (storm_df[storm_df.apply(tuple, axis=1) == nlf_tuple]).index.to_list()[0]

                # Crop the storm_df dataframe using this index
                cropped_storm_df = storm_df.loc[:nlf_index]
                facecolor, edgecolor, _ = categorize_landfall(nlf, plot_type='gen_to_lf')

                # Plot lysis point
                ax.scatter(nlf.lon, nlf.lat, s=s, facecolor=facecolor, marker='X',
                           edgecolor=edgecolor, zorder=5, linewidths=lw, alpha=a)
                # Plot storm track
                ax.plot(cropped_storm_df.lon.values, cropped_storm_df.lat.values, lw=lw*2, color=edgecolor, zorder=-1)

# Plots storm tracks based on a user-defined storm landfall region
def lf_to_gen(ax, storms, landfalls=None, nonlandfalls=None, lf=True, nlf=True, s=2.5, lw=0.5, a=1, div=4):
    if lf:
        lf_inds = range(len(landfalls))
        # Find average number of landfalls per ensemble member
        try:
            avg_num_lf = int(np.round((len(lf_inds)/len(np.unique(landfalls.index.get_level_values('enum'))))/div, 0))
        except:
            avg_num_lf = len(lf_inds)

        # Randomly subset landfalls to plot based on avg_num_lf   
        random_subset = np.random.choice(lf_inds, avg_num_lf, replace=False)
        temp_df = []
            
        for i in random_subset:
            # Extract landfall/storm data from parent dataframes
            temp_df.append(landfalls.iloc[[i]])
        
        landfalls = pd.concat(temp_df)
            
        for enum in np.unique(landfalls.index.get_level_values('enum')):
            ens_lfs = (landfalls.xs(enum, level='enum', drop_level=False)).reset_index()
            ens_stms = (storms.xs(enum, level='enum', drop_level=False)).reset_index()
                
            for index,lf in ens_lfs.iterrows():
                sid = lf.sid
                storm_df = ens_stms[ens_stms.sid == sid]
                
                # Make landfall data into a tuple
                lf_tuple = tuple(lf)
                
                # Find index in storm_df corresponding to landfall
                lf_index = (storm_df[storm_df.apply(tuple, axis=1) == lf_tuple]).index.to_list()[0]

                # Crop the storm_df dataframe using this index
                cropped_storm_df = storm_df.loc[:lf_index]
                facecolor, edgecolor, _ = categorize_landfall(lf, plot_type='lf_to_gen')
                    
                # Plot genesis point
                ax.scatter(lf.gen_lon, lf.gen_lat, s=s, facecolor=facecolor,
                           edgecolor=edgecolor, zorder=5, linewidths=lw, alpha=a)
                # Plot storm track
                ax.plot(cropped_storm_df.lon.values, cropped_storm_df.lat.values, lw=lw*2,
                        color=edgecolor,zorder=-1)

                   

    if nlf:
        nlf_inds = range(len(nonlandfalls))
            
        # Find average number of nonlandfalls per ensemble member
        try:
            avg_num_nlf = int(np.round((len(nlf_inds)/len(np.unique(nonlandfalls.index.get_level_values('enum'))))/div, 0))
        except:
            avg_num_nlf = len(lf_inds)

        # Randomly subset nonlandfalls to plot based on avg_num_lf   
        random_subset = np.random.choice(nlf_inds, avg_num_nlf, replace=False)

        temp_df = []
            
        for i in random_subset:
            # Extract nonlandfall data from parent dataframe
            temp_df.append(nonlandfalls.iloc[[i]])
                
        nonlandfalls = pd.concat(temp_df)
            
        for enum in np.unique(nonlandfalls.index.get_level_values('enum')):
            ens_nlfs = (nonlandfalls.xs(enum, level='enum', drop_level=False)).reset_index()
            ens_stms = (storms.xs(enum, level='enum', drop_level=False)).reset_index()  
            
            for index,nlf in ens_nlfs.iterrows():
                sid = nlf.sid
                storm_df = ens_stms[ens_stms.sid == sid]
                
                # Make landfall data into a tuple
                nlf_tuple = tuple(nlf)
                
                # Find index in storm_df corresponding to landfall
                nlf_index = (storm_df[storm_df.apply(tuple, axis=1) == nlf_tuple]).index.to_list()[0]

                # Crop the storm_df dataframe using this index
                cropped_storm_df = storm_df.loc[:nlf_index]
                facecolor, edgecolor, _ = categorize_landfall(nlf, plot_type='lf_to_gen')
                    
                # Plot lysis point
                ax.scatter(nlf.gen_lon, nlf.gen_lat, s=s, facecolor=facecolor, marker='X',
                           edgecolor=edgecolor, zorder=5, linewidths=lw, alpha=a)
                # Plot storm track
                ax.plot(cropped_storm_df.lon.values, cropped_storm_df.lat.values, lw=lw*2, color=edgecolor, zorder=-1)
                
# Create a 2-D track density histogram
def track_density(storms, bins=(np.arange(-120,0,4), np.arange(0,70,4))):
    h, x, y = np.histogram2d(storms.lon.values,storms.lat.values,bins=bins)
    return h, x, y

# Plot landfall points in a euclidean space relative to their genesis location centered at (0,0). 
# Storm duration is shown by scatter point size and intensity is shown by a colormap.
def intensity_duration(ax, storms, landfalls, num_lf=None, cmap='inferno', region='GL', vmin=900, vmax=1010,
                       scale=1, edgecolor='k', linewidth=0.25, alpha=0.6, intensity_metric='pressure',
                       nan_value=0, scatter=True, heatmap=True):
    xdiffs = []
    ydiffs = []
    life = []
    intensity = []
    
    # Subset landfalls by region
    landfalls = util.subset_storms(region,landfalls)
    lf_inds = range(len(landfalls))
    
    # Calculate average number of landfalls that a model produces over a set period (based on ensemble member)
    try:
        avg_num_lf = int(np.round(len(lf_inds)/len(np.unique(landfalls.index.get_level_values('enum'))),0))
    except:
        avg_num_lf = len(lf_inds)

    if num_lf is not None:
        avg_num_lf = num_lf
    
    # Randomly subset landfalls to plot based on avg_num_lf
    random_subset = np.random.choice(lf_inds,avg_num_lf,replace=False)
    
    for i in random_subset:
        # Extract landfall/storm data from parent dataframes
        landfall_df = landfalls.iloc[[i]]
        sid = landfall_df.index.get_level_values('sid')[0]
        storm_df = storms.xs(sid,level='sid')
        gen_lon = storm_df.gen_lon[0]
        gen_lat = storm_df.gen_lat[0]
        
        xdiffs.append(*(landfall_df.lon.values-landfall_df.gen_lon.values))
        ydiffs.append(*(landfall_df.lat.values-landfall_df.gen_lat.values))

        # Calculate storm lifetime
        try:
            dates = len(np.unique(pd.to_datetime(storm_df.index.get_level_values('time').date)))
        except:
            dates = math.ceil(len(storm_df.HOUR.values)/24)
        life.append(dates)
        
        if intensity_metric == 'pressure':
            # If pressure data is missing, convert wind to pressure
            if (storm_df.pressure == nan_value).all():
                # Coefficients for wind-pressure quadratic relationship
                a,b,c = -8.06310272e-03, -7.83002512e-01,  1.02042887e+03
                lf_wind = landfall_df.wind.values[0]
                wind_derived_pressure = a*(lf_wind)**2 + b*(lf_wind) + c
                intensity.append(wind_derived_pressure)

            else:
                intensity.append(landfall_df.pressure.values[0])
                
        elif intensity_metric == 'wind':
            intensity.append(landfall_df.wind.values[0])

    
    xdiffs_all = []
    ydiffs_all = []
    
    # Use complete storm data instead of random subset to calculate statistics
    for i in lf_inds:
        landfall_df = landfalls.iloc[[i]]
        sid = landfall_df.index.get_level_values('sid')[0]
        storm_df = storms.xs(sid,level='sid')
        gen_lon = storm_df.gen_lon[0]
        gen_lat = storm_df.gen_lat[0]
        
        xdiffs_all.append(*(landfall_df.lon.values-landfall_df.gen_lon.values))
        ydiffs_all.append(*(landfall_df.lat.values-landfall_df.gen_lat.values))
            
            
    count = [0,0,0,0]
    coords = list(zip(xdiffs_all ,ydiffs_all))
    
    # Compute fraction of storms that make landfall in each quadrant relative to genesis point
    for coord in coords:
        if (coord[0] > 0) & (coord[1] > 0): count[0] +=1
        elif (coord[0] < 0) & (coord[1] > 0): count[1] +=1
        elif (coord[0] < 0) & (coord[1] < 0): count[2] +=1
        else: count[3] +=1
    
    pm = None
    if scatter:
        # ax.set_facecolor('white')
        pm = ax.scatter(xdiffs, ydiffs, s=np.multiply(life, scale), edgecolor=edgecolor, linewidth=linewidth,
                        c=intensity, cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha)
        
    if heatmap:
        lon_bin_size = 8 
        lat_bin_size = 8
        vmax = 4/2.5 * lon_bin_size
        
        # ax.set_facecolor('gainsboro')
        lon_bins = np.arange(-100, 100 + lon_bin_size, lon_bin_size)
        lat_bins = np.arange(-50, 50 + lat_bin_size, lat_bin_size)
        h, x, y = np.histogram2d(xdiffs_all, ydiffs_all, bins=[lon_bins, lat_bins])
        h /= np.sum(h)
        hplot = np.where(h > 0, h, np.nan)
        print(np.nanpercentile(hplot, 95))
        pm = ax.pcolormesh(x, y, hplot.T * 100, cmap=cmap, vmin=0, vmax=vmax)
    
    return pm, count

# Decorate a plot to look more "map-like"
def mappy(ax, extent=[-100,-10,10,60], lat_interval=15, lon_interval=30, crs=ccrs.PlateCarree(), fontsize='x-small',
          borders='black', states_color='#FFFFFF00', state_borders='black', ocean_color='#FFFFFF00', land_color='#FFFFFF00'):
    # Add extent    
    lonW, lonE, latS, latN = extent[0], extent[1], extent[2], extent[3]
    # Add lat/lon ticks
    ax.set_yticks(np.arange(latS+lat_interval, latN, lat_interval), crs=crs, minor=False)
    ax.set_yticks(np.arange(latS, latN+lat_interval, lat_interval*2), crs=crs, minor=True)
    
    ax.set_xticks(np.arange(lonW+lon_interval, lonE, lon_interval), crs=crs, minor=False)
    ax.set_xticks(np.arange(lonW, lonE+lon_interval, lon_interval*2), crs=crs, minor=True)
    
    lon_labels = np.arange(lonW+lon_interval, lonE, lon_interval)
    lat_labels = np.arange(latS+lat_interval, latN, lat_interval)
    
    ax.set_xticklabels(lon_labels, fontsize=fontsize)
    ax.set_yticklabels(lat_labels, fontsize=fontsize)
    
    lon_formatter = LongitudeFormatter(zero_direction_label=True, number_format='.0f')
    lat_formatter = LatitudeFormatter(number_format='.0f')
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    
    ax.set_extent(extent,crs=ccrs.PlateCarree())
    
    ax.add_feature(cfeature.LAND.with_scale('10m'), facecolor=land_color, edgecolor=borders, linewidth=0.2)
    ax.add_feature(cfeature.OCEAN.with_scale('10m'), facecolor=ocean_color, edgecolor=borders, linewidth=0.2)
    ax.add_feature(cfeature.BORDERS, facecolor='none', edgecolor=borders, linewidth=.2)
    ax.add_feature(cfeature.COASTLINE, facecolor='none', edgecolor=borders, linewidth=.2)
    ax.add_feature(cfeature.STATES, facecolor=states_color, edgecolor=state_borders, linewidth=.2)
    ax.add_feature(cfeature.LAKES.with_scale('110m'), edgecolor=borders, facecolor=ocean_color, linewidth=.2)

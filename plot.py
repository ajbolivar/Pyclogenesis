import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from collections import Counter
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from Pyclogenesis import util

def storm_track_plt(ax,storms,storm_numbers,landfalls,size=1.2,plot_type='wind_speed',
                    scattercolor='blue',linecolor='gainsboro',linewidth=0.25,legend=False,
                    legend_colors=None,track_labels=None):
    ''' Plots storm tracks, wind speed, and landfall points (if applicable) by location.
        Parameters: 
	    ax (int arr): axis to draw plot on.
	    storm_numbers (int arr): number of storm tracks to plot (if all, specify length)
        landfalls (pandas.DataFrame):
	    size (float):
	    plot_type (str):
	    scattercolor (str):
	    linecolor (str):
	    linewidth (float):
	    legend (bool):
	    legend_colors (str arr):
	    track_labels (str arr):
            
        Returns:
	    storm_number_id (str):
            
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
                
        else:
            if len(storms) < 5:
                print("Storm {} did not make landfall.".format(storm_number_id))
                
    # Return SID if only plotting one storm
    if len(storm_numbers) == 1:
        return storm_number_id

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
        genesis_point = Point(landfall.GEN_LON, landfall.GEN_LAT)
        if mdr.contains(genesis_point): return 'mediumvioletred', 'deeppink', 'mdr'
        elif ws.contains(genesis_point): return 'darkblue', 'dodgerblue', 'ws'
        elif gm.contains(genesis_point): return 'darkgreen', 'limegreen', 'gm'
        elif cs.contains(genesis_point): return 'darkgoldenrod', 'gold', 'cs'
        elif natl.contains(genesis_point): return 'darkgray', 'gainsboro', 'other_natl'
        else: return 'none', 'none', 'outside'
             
    if plot_type == 'lf_to_gen':
        landfall_point = Point(landfall.LON, landfall.LAT)
        if gc.contains(landfall_point): return 'darkred', 'red', 'gc'
        elif se.contains(landfall_point): return 'mediumblue', 'deepskyblue', 'se'
        elif ne.contains(landfall_point): return 'rebeccapurple', 'darkviolet', 'ne'
        elif ci_ca.contains(landfall_point): return 'saddlebrown', 'darkorange', 'ci_ca'
        elif natl.contains(landfall_point): return 'darkgray', 'gainsboro', 'other_natl'
        else: return 'none','none', 'outside'

def spaghetti_table(landfalls=None, nonlandfalls=None, region='NA', plot_type='gen_to_lf', lf=True, nlf=True):
    '''Records count information for genesis and landfall regions.
    Parameters:
        landfalls (pandas DataFrame):
        nonlandfalls (pandas DataFrame):
        region (str):
        plot_type (str):
        lf (bool):
        nlf (bool):
    '''
    subregions = []
    if lf:
        landfalls = util.subset_storms(region, landfalls)
        for enum in np.unique(landfalls.index.get_level_values('ENUM')):
            ens_lfs = (landfalls.xs(enum,level='ENUM',drop_level=False)).reset_index()    
            for index, lf in ens_lfs.iterrows():
                _, _,subregion = categorize_landfall(lf, plot_type=plot_type)
                subregions.append(subregion)
    if nlf:
        for enum in np.unique(nonlandfalls.index.get_level_values('ENUM')):
            ens_nlfs = (nonlandfalls.xs(enum,level='ENUM',drop_level=False)).reset_index()
            for index, nlf in ens_nlfs.iterrows():
                _, _,subregion = categorize_landfall(nlf, plot_type=plot_type)
                subregions.append(subregion)


    subregions_count = Counter(subregions)
    return subregions_count

def gen_to_lf(ax,storms,landfalls=None,nonlandfalls=None,lf=True,nlf=True,s=2.5,lw=0.5,a=1,div=4):
    ''' Plots scatter points based on a user-defined storm genesis region.
    Parameters: 
        ax (arr):
        storms (pandas.DataFrame)
        landfalls (pandas.DataFrame):
        nonlandfalls (pandas.DataFrame)
        lf (bool):
        nlf (bool):
        s (float):
        lw (float):
        a (float):
        div (int)
    Returns:
        None
    '''

    subregions = []
    
    if lf:
        lf_inds = range(len(landfalls))
        # Find average number of landfalls per ensemble member
        try:
            avg_num_lf = int(np.round((len(lf_inds)/len(np.unique(landfalls.index.get_level_values('ENUM'))))/div,0))
        except:
            avg_num_lf = len(lf_inds)

        # Randomly subset landfalls to plot based on avg_num_lf   
        random_subset = np.random.choice(lf_inds,avg_num_lf,replace=False)
        temp_df = []
            
        for i in random_subset:
            # Extract landfall/storm data from parent dataframes
            temp_df.append(landfalls.iloc[[i]])
        
        landfalls = pd.concat(temp_df)
        
        for enum in np.unique(landfalls.index.get_level_values('ENUM')):
            ens_lfs = (landfalls.xs(enum,level='ENUM',drop_level=False)).reset_index()
            ens_stms = (storms.xs(enum,level='ENUM',drop_level=False)).reset_index()
            
            for index, lf in ens_lfs.iterrows():
                sid = lf.SID
                storm_df = ens_stms[ens_stms.SID == sid]
                
                # Make landfall data into a tuple
                lf_tuple = tuple(lf)
                
                # Find index in storm_df corresponding to landfall
                lf_index = (storm_df[storm_df.apply(tuple, axis=1) == lf_tuple]).index.to_list()[0]

                # Crop the storm_df dataframe using this index
                cropped_storm_df = storm_df.loc[:lf_index]
                facecolor, edgecolor, _ = categorize_landfall(lf, plot_type='gen_to_lf')

                # Plot genesis point
                ax.scatter(lf.LON, lf.LAT, s=s, facecolor=facecolor,
                           edgecolor=edgecolor, zorder=5, linewidths=lw, alpha=a)
                # Plot storm track
                ax.plot(cropped_storm_df.LON.values, cropped_storm_df.LAT.values, lw=lw*2, color=edgecolor, zorder=-1)

            
        
    if nlf:
        nlf_inds = range(len(nonlandfalls))
        # Find average number of nonlandfalls per ensemble member
        try:
            avg_num_nlf = int(np.round((len(nlf_inds)/len(np.unique(nonlandfalls.index.get_level_values('ENUM'))))/div,0))
        except:
            avg_num_nlf = len(lf_inds)

        # Randomly subset nonlandfalls to plot based on avg_num_lf   
        random_subset = np.random.choice(nlf_inds,avg_num_nlf,replace=False)
        temp_df = []
            
        for i in random_subset:
            # Extract nonlandfall data from parent dataframe
            temp_df.append(nonlandfalls.iloc[[i]])
                
        nonlandfalls = pd.concat(temp_df)
        for enum in np.unique(nonlandfalls.index.get_level_values('ENUM')):
            ens_nlfs = (nonlandfalls.xs(enum,level='ENUM',drop_level=False)).reset_index()
            ens_stms = (storms.xs(enum,level='ENUM',drop_level=False)).reset_index()
                
            for index,nlf in ens_nlfs.iterrows():
                sid = nlf.SID
                storm_df = ens_stms[ens_stms.SID == sid]
                
                # Make landfall data into a tuple
                nlf_tuple = tuple(nlf)
                
                # Find index in storm_df corresponding to landfall
                nlf_index = (storm_df[storm_df.apply(tuple, axis=1) == nlf_tuple]).index.to_list()[0]

                # Crop the storm_df dataframe using this index
                cropped_storm_df = storm_df.loc[:nlf_index]
                facecolor, edgecolor, _ = categorize_landfall(nlf,plot_type='gen_to_lf')

                # Plot lysis point
                ax.scatter(nlf.LON,nlf.LAT,s=s,facecolor=facecolor,marker='X',
                           edgecolor=edgecolor,zorder=5,linewidths=lw,alpha=a)
                # Plot storm track
                ax.plot(cropped_storm_df.LON.values,cropped_storm_df.LAT.values,lw=lw*2,color=edgecolor,zorder=-1)

            
def lf_to_gen(ax,storms,landfalls=None,nonlandfalls=None,lf=True,nlf=True,s=2.5,lw=0.5,a=1,div=4):
    ''' Plots scatter points based on a user-defined storm landfall region.
    Parameters: 
        ax (arr):
        storms (pandas.DataFrame)
        landfalls (pandas.DataFrame):
        nonlandfalls (pandas.DataFrame)
        lf (bool):
        nlf (bool):
        s (float):
        lw (float):
        a (float):
        div (int)
    Returns:
        None
    '''
    
    # try:
    if lf:
        lf_inds = range(len(landfalls))
        # Find average number of landfalls per ensemble member
        try:
            avg_num_lf = int(np.round((len(lf_inds)/len(np.unique(landfalls.index.get_level_values('ENUM'))))/div,0))
        except:
            avg_num_lf = len(lf_inds)

        # Randomly subset landfalls to plot based on avg_num_lf   
        random_subset = np.random.choice(lf_inds,avg_num_lf,replace=False)
        temp_df = []
            
        for i in random_subset:
            # Extract landfall/storm data from parent dataframes
            temp_df.append(landfalls.iloc[[i]])
        
        landfalls = pd.concat(temp_df)
            
        for enum in np.unique(landfalls.index.get_level_values('ENUM')):
            ens_lfs = (landfalls.xs(enum,level='ENUM',drop_level=False)).reset_index()
            ens_stms = (storms.xs(enum,level='ENUM',drop_level=False)).reset_index()
                
            for index,lf in ens_lfs.iterrows():
                sid = lf.SID
                storm_df = ens_stms[ens_stms.SID == sid]
                
                # Make landfall data into a tuple
                lf_tuple = tuple(lf)
                
                # Find index in storm_df corresponding to landfall
                lf_index = (storm_df[storm_df.apply(tuple, axis=1) == lf_tuple]).index.to_list()[0]

                # Crop the storm_df dataframe using this index
                cropped_storm_df = storm_df.loc[:lf_index]
                facecolor, edgecolor, _ = categorize_landfall(lf,plot_type='lf_to_gen')
                    
                # Plot genesis point
                ax.scatter(lf.GEN_LON,lf.GEN_LAT,s=s,facecolor=facecolor,
                           edgecolor=edgecolor,zorder=5,linewidths=lw,alpha=a)
                # Plot storm track
                ax.plot(cropped_storm_df.LON.values,cropped_storm_df.LAT.values,lw=lw*2,
                        color=edgecolor,zorder=-1)

                   

    if nlf:
        nlf_inds = range(len(nonlandfalls))
            
        # Find average number of nonlandfalls per ensemble member
        try:
            avg_num_nlf = int(np.round((len(nlf_inds)/len(np.unique(nonlandfalls.index.get_level_values('ENUM'))))/div,0))
        except:
            avg_num_nlf = len(lf_inds)

        # Randomly subset nonlandfalls to plot based on avg_num_lf   
        random_subset = np.random.choice(nlf_inds,avg_num_nlf,replace=False)

        temp_df = []
            
        for i in random_subset:
            # Extract nonlandfall data from parent dataframe
            temp_df.append(nonlandfalls.iloc[[i]])
                
        nonlandfalls = pd.concat(temp_df)
            
        for enum in np.unique(nonlandfalls.index.get_level_values('ENUM')):
            ens_nlfs = (nonlandfalls.xs(enum,level='ENUM',drop_level=False)).reset_index()
            ens_stms = (storms.xs(enum,level='ENUM',drop_level=False)).reset_index()  
            
            for index,nlf in ens_nlfs.iterrows():
                sid = nlf.SID
                storm_df = ens_stms[ens_stms.SID == sid]
                
                # Make landfall data into a tuple
                nlf_tuple = tuple(nlf)
                
                # Find index in storm_df corresponding to landfall
                nlf_index = (storm_df[storm_df.apply(tuple, axis=1) == nlf_tuple]).index.to_list()[0]

                # Crop the storm_df dataframe using this index
                cropped_storm_df = storm_df.loc[:nlf_index]
                facecolor, edgecolor, _ = categorize_landfall(nlf,plot_type='lf_to_gen')
                    
                # Plot lysis point
                ax.scatter(nlf.GEN_LON,nlf.GEN_LAT,s=s,facecolor=facecolor,marker='X',
                           edgecolor=edgecolor,zorder=5,linewidths=lw,alpha=a)
                # Plot storm track
                ax.plot(cropped_storm_df.LON.values,cropped_storm_df.LAT.values,lw=lw*2,color=edgecolor,zorder=-1)
                
                
def track_density(storms,bins=(np.arange(-120,0,4), np.arange(0,70,4))):
    ''' Creates a 2D track density histogram
    Parameters: 
        storms (pandas.DataFrame):
        bins (int or arr)):
    Returns:
        plot:
    '''
    h,x,y = np.histogram2d(storms.LON.values,storms.LAT.values,bins=bins)
    return h, x, y


def intensity_duration(ax,storms,landfalls,num_lf=None,cmap='inferno',region='GL',vmin=900,vmax=1010,
                       scale=1,edgecolor='k',linewidth=0.25,alpha=0.6,intensity_metric='pressure',
                       nan_value=0):
    ''' Plots landfall points in a euclidean space relative to their genesis location centered at (0,0). 
        Storm duration is shown by scatter point size and intensity is shown by a colormap.
        Parameters: 
            ax (arr):
            storms (pandas.DataFrame):
            landfalls (pandas.DataFrame):
            cmap (str): colormap name.
            vmin (str): minimum value for colormap.
            vmax (float): maximum value for colormap.
            edgecolor (str): color of scatter point edge.
            linewidth (float): width of scatter point edge.
            alpha (float): transparency of scatter point (1 for opaque, 0 for transparent).
            intensity_metric (string): metric used to represent storm intensity (options: 'wind','pressure'; default = 'pressure')
            
        Returns:
            pm ():
            count (int):
    '''
    
    xdiffs = []
    ydiffs = []
    life = []
    intensity = []
    
    # Subset landfalls by region
    landfalls = util.subset_storms(region,landfalls)
    lf_inds = range(len(landfalls))
    
    # Calculate average number of landfalls that a model produces over a set period (based on ensemble member)
    try:
        avg_num_lf = int(np.round(len(lf_inds)/len(np.unique(landfalls.index.get_level_values('ENUM'))),0))
    except:
        avg_num_lf = len(lf_inds)

    if num_lf is not None:
        avg_num_lf = num_lf
    
    # Randomly subset landfalls to plot based on avg_num_lf
    random_subset = np.random.choice(lf_inds,avg_num_lf,replace=False)
    
    for i in random_subset:
        # Extract landfall/storm data from parent dataframes
        landfall_df = landfalls.iloc[[i]]
        sid = landfall_df.index.get_level_values('SID')[0]
        storm_df = storms.xs(sid,level='SID')
        gen_lon = storm_df.GEN_LON[0]
        gen_lat = storm_df.GEN_LAT[0]
        
        xdiffs.append(*(landfall_df.LON.values-landfall_df.GEN_LON.values))
        ydiffs.append(*(landfall_df.LAT.values-landfall_df.GEN_LAT.values))
        
        # Calculate storm lifetime
        try:
            dates = len(np.unique(storm_df.index.get_level_values('ISO_TIME').date))
        except:
            dates = math.ceil(len(storm_df.HOUR.values)/24)
        life.append(dates)
        
        if intensity_metric == 'pressure':
            # If pressure data is missing, convert wind to pressure
            if (storm_df.PRES == nan_value).all():
                # Coefficients for wind-pressure quadratic relationship
                a,b,c = -8.06310272e-03, -7.83002512e-01,  1.02042887e+03
                lf_wind = landfall_df.WIND.values[0]
                wind_derived_pressure = a*(lf_wind)**2 + b*(lf_wind) + c
                intensity.append(wind_derived_pressure)

            else:
                intensity.append(landfall_df.PRES.values[0])
                
        elif intensity_metric == 'wind':
            intensity.append(landfall_df.WIND.values[0])

    
    xdiffs_all = []
    ydiffs_all = []
    
    # Use complete storm data instead of random subset to calculate statistics
    for i in lf_inds:
        landfall_df = landfalls.iloc[[i]]
        sid = landfall_df.index.get_level_values('SID')[0]
        storm_df = storms.xs(sid,level='SID')
        gen_lon = storm_df.GEN_LON[0]
        gen_lat = storm_df.GEN_LAT[0]
        
        xdiffs_all.append(*(landfall_df.LON.values-landfall_df.GEN_LON.values))
        ydiffs_all.append(*(landfall_df.LAT.values-landfall_df.GEN_LAT.values))
            
            
    count = [0,0,0,0]
    coords = list(zip(xdiffs_all ,ydiffs_all))
    
    # Compute fraction of storms that make landfall in each quadrant relative to genesis point
    for coord in coords:
        if (coord[0] > 0) & (coord[1] > 0): count[0] +=1
        elif (coord[0] < 0) & (coord[1] > 0): count[1] +=1
        elif (coord[0] < 0) & (coord[1] < 0): count[2] +=1
        else: count[3] +=1
    
    point_size = [20*l for l in life]
    
    pm = ax.scatter(xdiffs,ydiffs,s=np.multiply(life,scale),edgecolor=edgecolor,linewidth=linewidth,
                    c=intensity,cmap=cmap,vmin=vmin,vmax=vmax,alpha=alpha)
    
    return pm, count
        
     
    
def regional_lf(ax,landfalls,landing_states,state_names,colors=['white'],size=4,lw=0.25,zorder=100):
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

                

def temporal_lf(ax,landfalls,label,size=5,lw=0.25,color='red',edgecolor='black',months=[1],a=0.8,marker='o'):
    ''' Plots landfalls color-coded by seasonal timing (pre-season, early season, peak season,
        late season, post season).
    Parameters: 
        ax:
        landfalls (pandas.DataFrame):
        size (float):
        lw (float):
        color (str)
        edgecolor (str):
        months (int arr):
        a (float):
        marker (str):
    Returns:
        None
    '''
    
    lf = landfalls[(landfalls.index.get_level_values('ISO_TIME')).month.isin(months)]
    
    ax.scatter(lf.LON,lf.LAT,c=color,edgecolor=edgecolor,linewidth=lw,s=size,zorder=2,label=label,alpha=a,marker=marker)
    
    
    
def mappy(ax,extent=[-100,-10,10,60],lat_interval=15,lon_interval=30,crs=ccrs.PlateCarree(),fontsize='x-small',
          borders='black',states_color='#FFFFFF00',state_borders='black',ocean_color='#FFFFFF00',land_color='#FFFFFF00'):
    ''' Decorates plot to look more "map-like".
        Parameters: 
            ax:
            extent (int arr):
            lat_interval (float):
            lon_interval (float):
            crs (cartopy projection):
            fontsize (str or float):
            borders (str):
            states_color (str):
            state_borders (str):
            ocean_color (str):
            land_color (str):
            
        Returns:
            None
    '''
    # Add extent    
    lonW, lonE, latS, latN = extent[0], extent[1], extent[2], extent[3]

    # Add lat/lon ticks
    ax.set_yticks(np.arange(latS+lat_interval,latN,lat_interval), crs=crs, minor=False)
    ax.set_yticks(np.arange(latS,latN+lat_interval,lat_interval*2), crs=crs, minor=True)
    
    ax.set_xticks(np.arange(lonW+lon_interval,lonE,lon_interval), crs=crs, minor=False)
    ax.set_xticks(np.arange(lonW,lonE+lon_interval,lon_interval*2), crs=crs, minor=True)
    
    lon_labels = np.arange(lonW+lon_interval,(lonE),lon_interval)
    lat_labels = np.arange(latS+lat_interval,(latN),lat_interval)
    
    ax.set_xticklabels(lon_labels,fontsize=fontsize)
    ax.set_yticklabels(lat_labels,fontsize=fontsize)
    
    lon_formatter = LongitudeFormatter(zero_direction_label=True,number_format='.0f')
    lat_formatter = LatitudeFormatter(number_format='.0f')
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    
    ax.set_extent(extent,crs=ccrs.PlateCarree())
    
    ax.add_feature(cfeature.LAND.with_scale('10m'),facecolor=land_color,edgecolor=borders,linewidth=0.2)
    ax.add_feature(cfeature.OCEAN.with_scale('10m'),facecolor=ocean_color,edgecolor=borders,linewidth=0.2)
    ax.add_feature(cfeature.BORDERS,facecolor='none',edgecolor=borders,linewidth=.2)
    ax.add_feature(cfeature.COASTLINE,facecolor='none',edgecolor=borders,linewidth=.2)
    ax.add_feature(cfeature.STATES,facecolor=states_color,edgecolor=state_borders,linewidth=.2)
    ax.add_feature(cfeature.LAKES.with_scale('110m'),edgecolor=borders,facecolor=ocean_color,linewidth=.2)

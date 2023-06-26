import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.io.shapereader as shpreader
import os
import shapely.geometry as shp_geom
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
                print(storm_landings)
                
        else:
            if len(storms) < 5:
                print("Storm {} did not make landfall.".format(storm_number_id))
                
    # Return SID if only plotting one storm
    if len(storm_numbers) == 1:
        return storm_number_id

    
def gen_to_lf(ax,bounds,landfalls=None,nonlandfalls=None,lf=True,nlf=True,s=2.5,
              gen_color_lf='blue',lf_color='limegreen',gen_color_nlf='yellow',nlf_color='red',
              lf_edge='black',nlf_edge='black',gen_edge='black',lw=0.5,a=1):

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
        if lf:
            for sid in landfalls.index.get_level_values('SID'):
                storm_df = landfalls.xs(sid,level='SID')
                GEN_LAT  = storm_df.GEN_LAT.values[0]
                GEN_LON  = storm_df.GEN_LON.values[0]
                
                # MDR: 10-20N, 80-20W
                gen_W,gen_E,gen_S,gen_N = bounds[0],bounds[1],bounds[2],bounds[3]
                
                lat_check = (GEN_LAT>gen_S) & (GEN_LAT<gen_N)
                lon_check = (GEN_LON>gen_W) & (GEN_LON<gen_E)

                if lat_check & lon_check:
                    # Plot genesis point
                    ax.scatter(GEN_LON,GEN_LAT,s=s,facecolor=gen_color_lf,edgecolor=gen_edge,
                               zorder=5,linewidths=lw,alpha=a)
                    # Plot landfall point
                    ax.scatter(storm_df.LON.values,storm_df.LAT.values,s=s,facecolor=lf_color,
                               edgecolor=lf_edge,zorder=5,linewidths=lw,alpha=a)
        
        if nlf:
            for sid in nonlandfalls.index.get_level_values('SID'):

                storm_df = nonlandfalls.xs(sid,level='SID')
                GEN_LAT  = storm_df.GEN_LAT.values[0]
                GEN_LON  = storm_df.GEN_LON.values[0]

                # MDR: 10-20N, 80-20W
                gen_W,gen_E,gen_S,gen_N = bounds[0],bounds[1],bounds[2],bounds[3]
                
                lat_check = (GEN_LAT>gen_S) & (GEN_LAT<gen_N)
                lon_check = (GEN_LON>gen_W) & (GEN_LON<gen_E)

                if lat_check & lon_check:
                    # Plot genesis point
                    ax.scatter(GEN_LON,GEN_LAT,s=s,facecolor=gen_color_nlf,edgecolor=gen_edge,
                               zorder=5,linewidths=lw,alpha=a)
                    # Plot lysis point
                    ax.scatter(storm_df.LON.values,storm_df.LAT.values,s=s,facecolor=nlf_color,
                               edgecolor=nlf_edge,zorder=5,linewidths=lw,alpha=a,marker='X')
    
    # If no points within given bounds are detected
    except:
        print('Model has no storm genesis points in the defined region.')
        return

            
def lf_to_gen(ax,bounds,landfalls=None,nonlandfalls=None,lf=True,nlf=True,s=2.5,
              gen_color_lf='blue',lf_color='limegreen',gen_color_nlf='yellow',nlf_color='red',
              lf_edge='black',nlf_edge='black',gen_edge='black',lw=0.5,a=1):
    ''' Plots scatter points based on a user-defined storm landfall region.
    Parameters: 
        ax (arr):
        landfalls (pandas.DataFrame):
        nonlandfalls (pandas.DataFrame):
        bounds (float): bounds of region box (lonW, lonE, latS, latN: lon range from -180 to 180, lat range from -90 to 90)
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
            for sid in landfalls.index.get_level_values('SID'):
                storm_df = landfalls.xs(sid,level='SID')
                LF_LAT  = storm_df.LAT.values[0]
                LF_LON  = storm_df.LON.values[0]
                
                lf_W,lf_E,lf_S,lf_N = bounds[0],bounds[1],bounds[2],bounds[3]

                lat_check = (LF_LAT>lf_S) & (LF_LAT<lf_N)
                lon_check = (LF_LON>lf_W) & (LF_LON<lf_E)
                
                if lat_check & lon_check:
                    # Plot genesis point
                    ax.scatter(storm_df.GEN_LON.values,storm_df.GEN_LAT.values,s=s,facecolor=gen_color_lf,
                               edgecolor=gen_edge,zorder=5,linewidths=lw,alpha=a)
                    # Plot landfall point
                    ax.scatter(LF_LON,LF_LAT,s=s,facecolor=lf_color,edgecolor=lf_edge,
                               zorder=5,linewidths=lw,alpha=a)
        
        if nlf:
            for sid in nonlandfalls.index.get_level_values('SID'):
                storm_df = nonlandfalls.xs(sid,level='SID')
                NLF_LAT  = storm_df.LAT.values[0]
                NLF_LON  = storm_df.LON.values[0]

                # MDR: 10-20N, 80-20W
                nlf_W,nlf_E,nlf_S,nlf_N = bounds[0],bounds[1],bounds[2],bounds[3]
                
                lat_check = (NLF_LAT>nlf_S) & (NLF_LAT<nlf_N)
                lon_check = (NLF_LON>nlf_W) & (NLF_LON<nlf_E)

                if lat_check & lon_check:
                    # Plot genesis point
                    ax.scatter(storm_df.GEN_LON.values,storm_df.GEN_LAT.values,s=s,facecolor=gen_color_nlf,
                               edgecolor=gen_edge,zorder=5,linewidths=lw,alpha=a)
                    # Plot lysis point
                    ax.scatter(NLF_LON,NLF_LAT,s=s,facecolor=nlf_color,edgecolor=nlf_edge,
                               zorder=5,linewidths=lw,alpha=a,marker='X')
                    
    
    # If no points within given bounds are detected
    except:
        print('Model has no storm landfall points in the defined region.')
        return

def track_density(storms,bins=(np.arange(-120,0,4), np.arange(0,70,4)),cmap='inferno'):
    ''' Plots a 2D histogram displaying track density
    Parameters: 
        ax (arr):
        storms (pandas.DataFrame):
        bins (int or arr)):
        cmap (str):
    Returns:
        plot:
    '''
    h,x,y = np.histogram2d(storms.LON.values,storms.LAT.values,bins=bins)
    return h, x, y

def intensity_duration(ax,storms,landfalls,cmap='inferno',region='GL',vmin=900,vmax=1010,
                            scale=1,edgecolor='k',linewidth=0.25,alpha=0.6):
    ''' Plots landfall points in a euclidean space relative to their genesis location centered at (0,0). 
        Storm duration is shown by scatter point size and intensity is shown by a colormap.
        Parameters: 
            ax (arr):
            storms (pandas.DataFrame):
            landfalls (pandas.DataFrame):
            cmap (str):
            vmin (str):
            vmax (float):
            edgecolor (str):
            linewidth (float):
            alpha (float):
            
        Returns:
            pm ():
    '''
    
    xdiffs = []
    ydiffs = []
    xdiffsna = []
    ydiffsna = []
    
    life = []
    lifena = []
    intensity = []
    
    # Subset landfalls by region
    landfalls = util.subset_storms(region,landfalls)
    lf_inds = range(len(landfalls))
    
    # Calculate average number of landfalls that a model produces over a set period (based on ensemble member)
    try:
        avg_num_lf = int(np.round(len(lf_inds)/len(np.unique(landfalls.index.get_level_values('ENUM'))),0))
    except:
        avg_num_lf = len(lf_inds)
    
    # Randomly subset landfalls to plot based on avg_num_lf
    random_subset = np.random.choice(lf_inds,avg_num_lf,replace=False)
    
    for i in random_subset:
        # Extract landfall/storm data from parent dataframes
        landfall_df = landfalls.iloc[[i]]
        sid = landfall_df.index.get_level_values('SID')[0]
        storm_df = storms.xs(sid,level='SID')
        gen_lon = storm_df.GEN_LON[0]
        gen_lat = storm_df.GEN_LAT[0]
        
        # Check for missing pressure information, plot seperately
        if storm_df.PRES.isna().all():
            xdiffsna.append(*(landfall_df.LON.values-landfall_df.GEN_LON.values))
            ydiffsna.append(*(landfall_df.LAT.values-landfall_df.GEN_LAT.values))
            
            # Calculate storm lifetime
            dates = np.unique(storm_df.index.get_level_values('ISO_TIME').date)
            lifena.append(len(dates))
        
        # 
        else:
            xdiffs.append(*(landfall_df.LON.values-landfall_df.GEN_LON.values))
            ydiffs.append(*(landfall_df.LAT.values-landfall_df.GEN_LAT.values))
            
            # Calculate storm lifetime
            dates = np.unique(storm_df.index.get_level_values('ISO_TIME').date)
            life.append(len(dates))
            intensity.append(landfall_df.PRES.values[0])

    
    xdiffs_all = []
    ydiffs_all = []
    
    # Using complete storm data instead of random subset to calculate statistics
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
    
    pm = ax.scatter(xdiffs,ydiffs,s=np.multiply(life,scale),edgecolor=edgecolor,linewidth=linewidth,
                    c=intensity,cmap=cmap,vmin=vmin,vmax=vmax,alpha=alpha)
    ax.scatter(xdiffsna,ydiffsna,s=np.multiply(lifena,scale),edgecolor=edgecolor,linewidth=linewidth,
               facecolor='gainsboro',alpha=alpha)
    
    ax.axhline(0,color='k',alpha=0.5,linewidth=0.5,linestyle='--')
    ax.axvline(0,color='k',alpha=0.5,linewidth=0.5,linestyle='--')
    ax.set_xlim([-100,100])
    ax.set_ylim([-40,40])
    
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
        pre_col (str):
        early_col (str):
        peak_col (str):
        post_col (str):
        edgecolor (str):
    Returns:
        None
    '''
    
    # Pre-season: March, April, May
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
    ax.set_yticks(np.arange(latS,latN+lat_interval,lat_interval), crs=crs, minor=False)
    ax.set_yticks(np.arange(latS,latN+lat_interval,lat_interval/2), crs=crs, minor=True)
    
    ax.set_xticks(np.arange(lonW,lonE+lon_interval,lon_interval), crs=crs, minor=False)
    ax.set_xticks(np.arange(lonW,lonE+lon_interval,lon_interval/2), crs=crs, minor=True)
    
    lon_labels = np.arange(lonW,(lonE+lon_interval),lon_interval)
    lat_labels = np.arange(latS,(latN+lat_interval),lat_interval)
    
    ax.set_xticklabels(lon_labels,fontsize=fontsize)
    ax.set_yticklabels(lat_labels,fontsize=fontsize)
    
    lon_formatter = LongitudeFormatter(zero_direction_label=True,number_format='.1f')
    lat_formatter = LatitudeFormatter(number_format='.1f')
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    
    ax.set_extent(extent,crs=ccrs.PlateCarree())
    
    ax.add_feature(cfeature.LAND.with_scale('10m'),facecolor=land_color,edgecolor=borders,linewidth=0.25)
    ax.add_feature(cfeature.OCEAN.with_scale('10m'),facecolor=ocean_color,edgecolor=borders,linewidth=0.25)
    ax.add_feature(cfeature.BORDERS,facecolor='none',edgecolor=borders,linewidth=.25)
    ax.add_feature(cfeature.COASTLINE,facecolor='none',edgecolor=borders,linewidth=.25)
    ax.add_feature(cfeature.STATES,facecolor=states_color,edgecolor=state_borders,linewidth=.25)
    ax.add_feature(cfeature.LAKES.with_scale('110m'),edgecolor=borders,facecolor=ocean_color,linewidth=.25)

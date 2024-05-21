import os
import time
import string
import math
import glob
import numpy as np
import xarray as xr
import pandas as pd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.patches as pat
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from Pyclogenesis import plot as tc
from Pyclogenesis import util

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

######################################################################################
# Polygons of regions for landfall/genesis region distributions (change as you see fit)
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

def region_pie(ax, landfalls, plot_type):
    """Creates pie chart embedded into spaghetti plot.
       Parameters:
       ax (int arr): axis to draw plot on.
       landfalls (pandas.DataFrame): DataFrame containing landfall information.
       plot_type (str): option to select whether landfalls are categorized by 
       landfall region or genesis region (options: 'gen_to_lf', 'lf_to_gen').
       
       Returns:
       None
    """
    
    lfs = landfalls

    if plot_type == 'gen_to_lf':
        na_count = sum([natl.contains(Point(lon, lat)) for lon, lat in zip(lfs.GEN_LON, lfs.GEN_LAT)])
        mdr_count = sum([mdr.contains(Point(lon, lat)) for lon, lat in zip(lfs.GEN_LON, lfs.GEN_LAT)])
        ws_count = sum([ws.contains(Point(lon, lat)) for lon, lat in zip(lfs.GEN_LON, lfs.GEN_LAT)])
        gm_count = sum([gm.contains(Point(lon, lat)) for lon, lat in zip(lfs.GEN_LON, lfs.GEN_LAT)])
        cs_count = sum([cs.contains(Point(lon, lat)) for lon, lat in zip(lfs.GEN_LON, lfs.GEN_LAT)])
        other_count = na_count - (mdr_count + ws_count + gm_count + cs_count)
        counts = [mdr_count, ws_count, gm_count, cs_count, other_count]
        colors = ['deeppink', 'dodgerblue', 'limegreen', 'gold', 'darkgray']
    elif plot_type == 'lf_to_gen':
        na_count    = sum([natl.contains(Point(lon,lat)) for lon,lat in zip(lfs.LON,lfs.LAT)])
        gc_count    = sum(gc.contains(Point(lon,lat)) for lon,lat in zip(lfs.LON,lfs.LAT))
        ne_count    = sum(ne.contains(Point(lon,lat)) for lon,lat in zip(lfs.LON,lfs.LAT))
        se_count    = sum(se.contains(Point(lon,lat)) for lon,lat in zip(lfs.LON,lfs.LAT))
        ci_ca_count = sum(ci_ca.contains(Point(lon,lat)) for lon,lat in zip(lfs.LON,lfs.LAT))
        other_count = na_count - (gc_count + ne_count + se_count + ci_ca_count)
        counts = [gc_count,ne_count,se_count,ci_ca_count,other_count]
        colors = ['red','darkviolet','deepskyblue','darkorange','darkgray']
        
    ax.pie(counts,colors=colors)
    circle = plt.Circle((0, 0), 1, color='black', edgecolor='black', linewidth=1, zorder=0)
    ax.add_artist(circle)

def spaghetti_plot(keys, filepath, dpi=300, rows=1, cols=1, panel_height=1, region='NA', s=2.0, lw=0.2,
                   a=1.0, lf=True, nlf=False, p_ax=[0.765, 0.7, 0.3, 0.3], plot_type='gen_to_lf',
                   fontsize=6, start_letter='a',extent=[-100,-10,10,55], lat_interval=15, lon_interval=30, 
                   borders='gray',states_color='gainsboro', state_borders='gray', 
                   ocean_color='#FFFFFF00', land_color='gainsboro',label_xy=(95,40)):

    """Creates and saves spaghetti plot of storm tracks for one or more products.
       Parameters:
       keys (str arr): list of keys to search for and identify files.
       filepath (str): filepath to save image to.
       dpi (int): dpi of output image.
       rows (int): number of rows for subplots. 
       cols (int): number of columns for subplots.
       panel_height (float): panel height in inches.
       s (float): size of scatter point.
       lw (float): width of scatter point edge.
       a (float): alpha of scatter point.
       lf (bool): when True, plots landfall points.
       nlf (bool): when True, plots lysis points.
       p_ax (float arr): coordinates and size of pie chart [x, y, width, height].
       plot_type (str): option to select whether landfalls are categorized by 
       landfall region or genesis region (options: 'gen_to_lf', 'lf_to_gen').
       fontsize (float or str): font size for plot labels.
       start_letter (str): letter to start from for panel annotations.
       extent (int arr): map extent shown in plot [lonE, lonW, latS, latN].
       lat_interval (int): interval of latitude ticks on y-axis.
       lon_interval (int): interval of longitude ticks on x-axis.
       borders (str): color of country borders.
       states_color (str): fill color of states.
       state_borders (str): color of state borders.
       ocean_color (str): fill color of oceans.
       land_color (str): fill color of land.
       
       Returns:
       None
    """
    # 
    panel_width = panel_height * ((extent[1] - extent[0]) / (extent[3] - extent[2]))
    left_margin, right_margin, top_margin, bot_margin = 0.5, 0.05, 0.05, 0.5

    figsize = ((panel_width * cols) + left_margin + right_margin, 
               (panel_height * rows) + top_margin + bot_margin)
    lmar, rmar = left_margin/figsize[0], right_margin/figsize[0]
    tmar, bmar = top_margin/figsize[1], bot_margin/figsize[1]
    
    fig,axes = plt.subplots(rows,cols,subplot_kw={'projection':ccrs.PlateCarree()},dpi=dpi,
                            figsize=figsize, sharex=True,sharey=True)

    if rows == 1 & cols == 1:
        axes = np.array(axes)

    print("Creating spaghetti plot...")
    start = time.time()
    for ax,key,letter in zip(axes.flat, keys, range(ord(start_letter), ord(start_letter) + len(keys))):
        if '-D' not in key:
            num_lf = len(np.unique(util.subset_storms(region, landfalls[key]).index.get_level_values('SID')))
        else:
            num_lf = len(np.unique(landfalls[key].index.get_level_values('SID')))
        # scalar by which the length of the dataset is divided to plot a subset of storms
        div = math.ceil(max(1,num_lf/100))

        if plot_type == 'gen_to_lf':
            tc.gen_to_lf(ax,storms=storms[key],landfalls=landfalls[key],lf=lf,nlf=nlf,lw=lw,a=a,s=s,div=div)
        elif plot_type == 'lf_to_gen':
            tc.lf_to_gen(ax,storms=storms[key],landfalls=landfalls[key],lf=lf,nlf=nlf,lw=lw,a=a,s=s,div=div)

        pie_ax = ax.inset_axes(p_ax)
        region_pie(pie_ax, landfalls[key], plot_type=plot_type)
        ax.annotate(f'({chr(letter)}) {key}', xy=(0.025, 0.95), xycoords='axes fraction',size=fontsize, ha='left', va='top',
                    bbox=dict(boxstyle='round',facecolor='white',linewidth=lw,edgecolor='k'), zorder=10)
        
        if rows != 1 & cols != 1:
            ax.tick_params(labelleft=ax in axes[:,0],left=ax in axes[:,0], 
                           labelbottom=ax in axes[-1,:],bottom=ax in axes[-1,:])

        tc.mappy(ax=ax,extent=extent,lat_interval=lat_interval,lon_interval=lon_interval,fontsize=fontsize,
                 borders=borders,states_color=states_color,state_borders=state_borders,ocean_color=ocean_color,
                 land_color=land_color)

    plt.subplots_adjust(wspace=0, hspace=0, left=lmar, right=1-rmar, top=1-tmar, bottom=bmar)

    end = time.time()
    plt.savefig(filepath,dpi=1000,bbox_inches='tight')
    print(f'Figure saved as {filepath}.\nTime elapsed: {end - start}')


def scatter_plot(keys, filepath, num_lf=None, region='NA', dpi=300, rows=1, cols=1, 
                 s=2, lw=0.15, a=0.75, div=1, fontsize=6, panel_height=1, start_letter='a', cmap='inferno',
                 vmin=900, vmax=1020, edgecolor='k', label_xy=(95,40)):

    """Creates and saves scatter plot of storm landfalls for one or more products.
       Parameters:
       keys (str arr): list of keys to search for and identify files.
       filepath (str): filepath to save image to.
       dpi (int): dpi of output image.
       rows (int): number of rows for subplots. 
       cols (int): number of columns for subplots.
       figsize (float tuple):
       s (float): size of scatter point.
       lw (float): width of scatter point edge.
       a (float): alpha of scatter point.
       fontsize (float): font size for plot labels.
       start_letter (str): letter to start from for panel annotations.
       cmap (str): colormap for pressure information.
       vmin (int): minimum value to display for pressure.
       vmax (int): maximum value to display for pressure.
       edgecolor (str): color of scatter point edge.
       
       Returns:
       None
    """

    panel_width = panel_height * 2
    left_margin, right_margin, top_margin, bot_margin = 0.5, 0.9, 0.05, 0.5

    figsize = ((panel_width * cols) + left_margin + right_margin, 
               (panel_height * rows) + top_margin + bot_margin)
    lmar, rmar = left_margin/figsize[0], right_margin/figsize[0]
    tmar, bmar = top_margin/figsize[1], bot_margin/figsize[1]
    
    fig,axes = plt.subplots(rows, cols, figsize=figsize, dpi=dpi, gridspec_kw = {'wspace':0,'hspace':0})
    
    if (rows == 1) & (cols == 1):
        axes = np.array(axes)

    print("Creating scatter plot...")
    start = time.time()
    for key,ax,letter in zip(keys, axes.flat, range(ord(start_letter), ord(start_letter) + len(keys))):
        ax.set_aspect(1)
        stm = storms[key]
        lf = landfalls[key]

        ax.set_facecolor='white'

        rad = [25, 50, 75, 100]
        t = np.linspace(0, 2*np.pi, 100)

        for r in rad:
            ax.plot(r*np.cos(t), r*np.sin(t), zorder=0, color='gainsboro', linewidth=0.3)

        ax.axhline(0, zorder=0, color='gainsboro', linewidth=0.3)
        ax.axvline(0, zorder=0, color='gainsboro', linewidth=0.3)
        pm, count = tc.intensity_duration(ax=ax, storms=stm, landfalls=lf, num_lf=num_lf, cmap=cmap,
                                          region=region, scale=s, vmin=vmin, vmax=vmax, 
                                          edgecolor=edgecolor, linewidth=lw, alpha=a)

        total = np.sum(count)
        
        ax.set_xlim([-100,100])
        ax.set_ylim([-50,50])

        ax.set_xticks(np.arange(-100,101,25))
        ax.set_xticklabels(['',-75,-50,-25,0,25,50,75,''])
        
        ax.set_yticks(np.arange(-50,51,25))
        ax.set_yticklabels(['',-25,0,25,''])
        
        ax.tick_params('both',labelsize=fontsize)

        if (rows != 1) & (cols != 1):
            ax.tick_params(labelleft=ax in axes[:,0], left=ax in axes[:,0], 
                           labelbottom=ax in axes[-1,:], bottom=ax in axes[-1,:])

        left, right, bottom, top = 0.05, 0.95, 0.08, 0.92 

        # Calculate the positions of the four corners relative to the subplot's coordinates
        corners = [(right, top), (left, top), (left, bottom), (right, bottom)]
        hor_align = ['right', 'left', 'left', 'right']
        ver_align = ['top', 'top', 'bottom', 'bottom']

        for corner, count_val, ha, va in zip(corners, count, hor_align, ver_align):
            ax.text(corner[0], corner[1], '{:.3f}'.format(count_val / total), 
                    transform=ax.transAxes, fontsize=fontsize, ha=ha, va=va)

        ax.annotate(f'({chr(letter)}) {key}', xy=(0.5, 0.05), xycoords='axes fraction',size=fontsize, ha='center', va='bottom',
                    bbox=dict(boxstyle='round',facecolor='white',linewidth=lw,edgecolor='k'))

    plt.subplots_adjust(wspace=0, hspace=0, left=lmar, right=1-rmar, top=1-tmar, bottom=bmar)

    if (rows != 1) & (cols != 1):
        cbar_bot = axes[-1,1].get_position().y0 # Bottom of the colorbar
        cbar_top = axes[0,1].get_position().y1 # Top of the colorbar
        cbar_left = axes[0,1].get_position().x1 # Left edge of the colorbar
        
    else:
        cbar_bot = ax.get_position().y0 # Bottom of the colorbar
        cbar_top = ax.get_position().y1 # Top of the colorbar
        cbar_left = ax.get_position().x1 # Left edge of the colorbar
    
    cax = fig.add_axes([cbar_left+.02, cbar_bot, 0.03, cbar_top - cbar_bot]) # x0, y0, width, height
    cbar = fig.colorbar(pm, cax = cax, extend='max') # shrink=0.5,pad=0.1,aspect=)
    cbar.ax.tick_params(labelsize = fontsize)

    end = time.time()
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
    print(f'Figure saved as {filepath}.\nTime elapsed: {end - start}')


def statistics_heatmap(keys, filepath, filetype='png', figsize=(3,4), dpi=500, region='NA', cmap='RdBu_r', fontsize=6.0, nan_value=-999, sdd=False):
    """Creates and saves heatmap containing statistics for one or more products.
       Parameters:
       keys (str arr):
       filepath (str):
       figsize (float tuple):
       dpi (int):
       cmap (str):
       fontsize (float):
       
       Returns:
       None
    """
    print("Creating statistics heatmap...")
    stats = {}
    start = time.time()
    for key in keys:
        print(f'Computing statistics for {key}...')
        if sdd == True:
            if key == 'IBTrACS': obs = True
            else: obs = False
            stats[key] = util.storm_statistics_sdd(storms[key], landfalls[key], nonlandfalls[key], name=key, obs=obs,
                                                   region=region, nan_value=-999, scale=0.152)
        else:
            stats[key] = util.storm_statistics(storms[key], landfalls[key], nonlandfalls[key], name=key, 
                                               region=region, nan_value=-999)

    # Combine all stats into one DataFrame
    stats_df = pd.concat(stats)
    stats_df = stats_df.set_index('NAME')
    pres_df = stats_df.iloc[:,-2:]
    stats_df = stats_df.drop(['AVG_LF_PRES','AVG_MIN_PRES'],axis=1)
    # Compute pressure biases separately
    try:
        pres_diffs = (pres_df-pres_df.loc['IBTrACS'])
    except:
        raise ValueError('IBTrACS is needed as a reference to produce an anomaly heatmap.')
        
    # Compute biases, scale by column
    stats_diffs = (stats_df - stats_df.loc['IBTrACS'])/stats_df.loc['IBTrACS']

    fig,ax = plt.subplots(1,1,figsize=figsize,dpi=dpi)

    df_shape = np.shape(stats_diffs)
    
    rows, cols = df_shape[0], df_shape[1]
    print(stats_diffs.shape)           
    pm=ax.pcolormesh(np.arange(0, cols, 1),
                     np.arange(rows, 0, -1),
                     stats_diffs.values,
                     cmap=cmap,vmax=1,vmin=-1)
    
    for i in range(cols):
        for j in range(rows):
            if np.abs(np.flip(np.array(stats_diffs),axis=0)[j,i]) >= 0.5:
                c = 'w'
            else:
                c = 'k'
 
            text = ax.text(i, j+1, np.flip(np.array(stats_df),axis=0)[j, i], fontsize=fontsize,
                           ha="center", va="center", color=c)
            
    stats_labels = stats_diffs.columns
    
    ax.set_xticks(np.arange(0, cols, 1))
    ax.set_xticklabels(stats_labels, fontweight='bold', fontsize=fontsize-1)
    ax.xaxis.set_label_position('top') 
    
    ax.set_yticks(np.arange(rows, 0, -1))
    ax.set_yticklabels(stats_diffs.index.values, fontweight='bold', fontsize=fontsize-1)
    ax.tick_params('x', labelrotation=90)
    ax.grid(which='minor', color='w')

    fig.subplots_adjust(top = 0.9)
    cbar_left = ax.get_position().x0 #Left edge of the colorbar
    cbar_right = ax.get_position().x1 #Right edge of the colorbar
    cbar_bot = ax.get_position().y1 #Bottom edge of the colorbar
    cbar_height_in = 0.25
    cax = fig.add_axes([cbar_left, cbar_bot+0.02, cbar_right-cbar_left, cbar_height_in / figsize[1]]) #x0, y0, width, height
    cbar = fig.colorbar(pm, cax=cax, orientation='horizontal', extend='both')
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.set_xticks(np.arange(-1,1.1,0.5), labels=['Low bias','','No bias','','High bias'], fontweight='bold')
    cbar.ax.tick_params(labelsize=fontsize)
    plt.savefig(f'{filepath}.{filetype}', dpi=dpi, bbox_inches="tight")

    ###############################################################################

    fig,ax = plt.subplots(1,1,figsize=(1,4),dpi=300)

    pm=ax.pcolormesh(np.arange(0, 2, 1),
                     np.arange(rows, 0, -1),
                     pres_diffs.values,
                     cmap='RdBu_r', vmax=20, vmin=-20)
    
    for i in range(len(pres_diffs.columns.values)):
        for j in range(len(pres_diffs.index.values)):
            if np.abs(np.flip(np.array(pres_diffs),axis=0)[j,i]) >= 10:
                c = 'w'
            else:
                c = 'k'
    
            text = ax.text(i, j+1, np.flip(np.array(pres_df),axis=0)[j, i], fontsize=fontsize,
                           ha="center", va="center", color=c)
            
    ax.set_xticks(np.arange(0,2,1))
    ax.set_xticklabels(['LF SLP','Min SLP'],fontweight='bold',fontsize=4)
    ax.xaxis.set_label_position('top') 
    ax.tick_params('x',labelrotation=90)
    ax.grid(which='minor',color='w')
    
    fig.subplots_adjust(right = 0.9)
    
    cbar_bot = ax.get_position().y0 #Bottom of the colorbar
    cbar_top = ax.get_position().y1 #Top of the colorbar
    cbar_left = ax.get_position().x1 #Left edge of the colorbar
    cax = fig.add_axes([cbar_left+.05, cbar_bot, 0.05, cbar_top - cbar_bot]) #x0, y0, width, height
    cbar = fig.colorbar(pm,cax = cax)#shrink=0.5,pad=0.1,aspect=)
    cbar.ax.tick_params(labelsize = 6)
    ax.set_yticks([])
    
    plt.savefig(f'{filepath}_pres.{filetype}', dpi=dpi, bbox_inches="tight")
    end = time.time()
    print(f'Figure saved as {filepath}_pres.{filetype}.\nTime elapsed: {end - start}')

def open_files(path, keys, labels, keyword='', icol=None):
    """Opens .csv files and stores them in a dictionary.
       Parameters:
       path (str): filepath to search for files.
       keys (str arr): list of keys to search for and identify files.
       keyword (str): optional parameter for additional keyword in files.
       icol (str arr): list of columns to make into indices.
       
       Returns:
       dictionary (pandas.DataFrame dict): dictionary containing DataFrames of data organized by supplied keys.
    """
    print(f'Opening the following files using the keyword \'{keyword}\'...')
    dictionary = {}
    for k, l in zip(keys, labels):
        filepath = glob.glob(f'{path}/{k}_*{keyword}*.csv')[0]
        print(f'\t{path}/{k}_*{keyword}.csv')
        file = pd.read_csv(filepath, index_col=icol, parse_dates=True)
        file.fillna(-999, inplace=True)
        dictionary[l] = file
    return dictionary

######################################################################################
# Edit filepath according to where your data is stored
input_dir  = '/glade/work/abolivar/Pyclogenesis_data/sdd_model_storms/monthly'
output_dir = '/glade/u/home/abolivar'
# List of product names (should be incorporated into file name)
products = ['ERA5']#,'ERA5','HadGEM3-GC31-HM', 'HadGEM3-GC31-LM', 'ECMWF-IFS-HR', 'ECMWF-IFS-LR',
            # 'EC-Earth3P-HR','EC-Earth3P','CNRM-CM6-1-HR','CNRM-CM6-1','MPI-ESM1-2-XR','MPI-ESM1-2-HR']
# Labels can be different from products for the purposes of plotting
labels   = ['ERA5-D']#,'ERA5','HadGEM3-GC31-HM', 'HadGEM3-GC31-LM', 'ECMWF-IFS-HR', 'ECMWF-IFS-LR',
            # 'EC-Earth3P-HR','EC-Earth3P','CNRM-CM6-1-HR','CNRM-CM6-1','MPI-ESM1-2-XR','MPI-ESM1-2-HR']
row_num = 1
col_num = 1

storms       = open_files(f'{input_dir}/all_storms', products, labels, 
                          keyword='storms', icol=['ENUM', 'SID'])
landfalls    = open_files(f'{input_dir}/landfalling_storms', products, labels, 
                          keyword='landfalls', icol=['ENUM', 'SID'])
nonlandfalls = open_files(f'{input_dir}/nonlandfalling_storms', products, labels, 
                          keyword='nonlandfalls', icol=['ENUM', 'SID'])

# To modify figure characteristics, please see parameters in the respective functions below
# The parameters rows and cols are set to 1 by default. If you have multiple products to plot, 
# please change the rows and cols parameters or it will simply plot the first one in your list 
spaghetti_plot(keys=labels, filepath=f'{output_dir}/spaghetti_gen2lf_era5-sdd.pdf', plot_type='gen_to_lf',
               rows=row_num, cols=col_num, start_letter='b')
spaghetti_plot(keys=labels, filepath=f'{output_dir}/spaghetti_lf2gen_era5-sdd.pdf', plot_type='lf_to_gen',
               rows=row_num, cols=col_num, start_letter='b')
scatter_plot(keys=labels, filepath=f'{output_dir}/scatter_era5-sdd.pdf', s=2, lw=0.25, 
             rows=row_num, cols=col_num, start_letter='b')
# IBTrACS is required to use this function
# statistics_heatmap(keys=labels, filepath=f'{output_dir}/stats_gcm',figsize=(4,4),filetype='pdf')

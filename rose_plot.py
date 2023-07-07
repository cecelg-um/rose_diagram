# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 11:34:40 2023

@author: cecelg

manupulating xarray data to plotly freq format
wind rose plots using plotly and xarray

"""

#%% import packages 

import numpy as np 
import xarray as xr
import pandas as pd

import plotly.io as pio
import plotly.express as px

from itertools import pairwise

#%% test plotly
# https://stackoverflow.com/questions/60742461/how-to-make-a-bar-polar-chart-with-plotly-without-frequency-column

print('plotting dummy / preformatted data from plotly...')

# test out sample data
df = px.data.wind()
fig = px.bar_polar(df, r="frequency", theta="direction",
                   color="strength", 
                   # template="plotly_dark",
                   title="dummy data",
                   color_discrete_sequence= px.colors.sequential.Plasma_r)
pio.renderers.default = 'browser'
fig.show()

del df
del fig

print('now will create with own data...')
print('will bin and compute frequency stats...')

#%% open xr data

# open_path = 'Z:/MODIS-LAADS-DAAC/CLG_GBB_Project/merged_data/18UTC_0.01Deg/'
open_path = 'Z:/MODIS-LAADS-DAAC/CLG_GBB_Project/scripts/github/'
# open_path = 'Z:/your_dowload_path/'
ds  = xr.open_dataset(open_path + 'Bahamas_Crop.nc', chunks={'time':50})

# open_path = 'Z:/MODIS-LAADS-DAAC/CLG_GBB_Project/merged_data/18UTC_0.01Deg/'
# ds  = xr.open_mfdataset(paths = open_path + 'merged_0.01Deg_18UTC_gbb_shp_small_v5.nc', chunks={'time':50})

#%% select data variablies

# ds vars to plot
var_mag = 'wind_mag'
var_dir = 'wind_dir'

# subset data - analyze every pixel
ds_sub = xr.merge([ds[var_mag], ds[var_dir]])
df_vars = ds_sub.to_dataframe(dim_order = None) # if memory / time allows

# create average - reduce spatial to one value for each timestep
# ds_mean = ds_sub.mean(['lat',  'lon'])
# df_vars = ds_mean.to_dataframe(dim_order = None)

#%% create bins

# magnitude histogram
fig = px.histogram(df_vars, x=var_mag)
pio.renderers.default = 'browser'   # use if IDE not web-enabled
fig.show()

# direction histogram
# fig = px.histogram(df_vars, x=var_dir)
# pio.renderers.default = 'browser'   # use if IDE not web-enabled
# fig.show()

# # autodetect bins
# mag_min = df_vars[var_mag].min()
# mag_max = df_vars[var_mag].max()
# mag_max_ceil  = np.ceil(mag_max)
# mag_max_floor = np.floor(mag_max)
# mag_bin = np.linspace(0, mag_max_ceil, num=10) # autodetect cutoffs; set bin #

# manually create bins
mag_bin = np.round(np.linspace(0, 14, num=8),2) # keep constant

#%% optimized rose function

def plotly_rose(df_vars):
    print('\n plotly rose function initiated...')
    print('\n using default direction bins')
    dir_dict = {
            'N'  : [348.75, 11.25],
            'NNE': [11.25, 33.75],
            'NE' : [33.75, 56.25],
            'ENE': [56.25, 78.75],
            'E'  : [78.75, 101.25],
            'ESE': [101.25, 123.75],
            'SE' : [123.75, 146.25],
            'SSE': [146.25, 168.75],
            'S'  : [168.75, 191.25],
            'SSW': [191.25, 213.75],
            'SW' : [213.75, 236.25],
            'WSW': [236.25, 258.75],
            'W'  : [258.75, 281.25],
            'WNW': [281.25, 303.75],
            'NW' : [303.75, 326.25],
            'NNW': [326.25, 348.75]
            }
    print(dir_dict)
    # create new direction and manitude bin column
    df_vars[var_dir+'2'] = None
    df_vars[var_mag+'2'] = None

    print('\n beginning direction binning ...')
    for sub_dir in dir_dict:
        # print(sub_dir)
        north_test = dir_dict[sub_dir][1]-dir_dict[sub_dir][0]
        if north_test > 0: 
            df_vars.loc[
                (df_vars[var_dir].between(
                    dir_dict[sub_dir][0], dir_dict[sub_dir][1])
                    ), [var_dir+'2']] = sub_dir
        else:
            # special case for north
            # not a cute way to solve
            df_vars.loc[
                (df_vars[var_dir].between(
                    dir_dict[sub_dir][0], 360)
                    ), [var_dir+'2']] = sub_dir
            df_vars.loc[
                (df_vars[var_dir].between(
                    0, dir_dict[sub_dir][1])
                    ), [var_dir+'2']] = sub_dir

    print('\n beginning magnitude binning ...')
    print('\n using default bins...')
    mag_bin = np.round(np.linspace(0, 14, num=8),2) # keep constant
    
    # find sig figs for bin padding 
    mag_bins = list(pairwise(mag_bin))
    max_bin_num= max(max(mag_bins))
    sig_dig = len(str(max_bin_num))

    for mg_bin in mag_bins:
        print("assigning magnitudes to bin: ", mg_bin)
        # correct for bins that are of dif sig figs
        low_str  = str(mg_bin[0]).rjust(sig_dig, '0')
        high_str = str(mg_bin[1]).rjust(sig_dig, '0')
        rng_str  = low_str + ' - ' + high_str
        df_vars.loc[
            (df_vars[var_mag].between(
                mg_bin[0], mg_bin[1])
                ), [var_mag+'2']] = rng_str

    print('\n establishing frequencies ...')
    # group observations into frequencies    
    grp = df_vars.groupby(
        [var_dir+'2', var_mag+'2']
        ).size().reset_index(name="frequency")
    
    # list of all possible dir given mag bins
    list_dir = []
    for k in range(len(mag_bins)):
        # print(k)
        for sub_dir in dir_dict:
            list_dir.append(sub_dir)
            
    # create list of all possible mag bins
    list_bins =[]
    for k in range(len(list(dir_dict.keys()))):
        # print(k)
        for mg_bin in mag_bins:
            low_str  = str(mg_bin[0]).rjust(sig_dig, '0')
            high_str = str(mg_bin[1]).rjust(sig_dig, '0')
            rng_str  = low_str + ' - ' + high_str
            list_bins.append(rng_str)
            
    # concat list of possible dis and mags into df 
    dir_series = pd.Series(list_dir, name = var_dir+'2')
    bin_series = pd.Series(list_bins, name = var_mag+'2')
    full_df = pd.concat([dir_series, bin_series], axis =1)
    full_df['frequency'] = 0    # freq = 0 unless matched

    # match obs data fequencies to all possible dir mag bins
    for idx, row in full_df.iterrows():
        # print(idx)
        # print("finding match for: ", row[0], row[1])
        for idx2, row2 in grp.iterrows():
        #     # print(row2[0])
        #     # print(row2[1])
            if (row2[0] == row[0]) == True:
                # print('dir_match... checking mag')
                if (row2[1] == row[1]) == True:
                    # print('mag match')
                    freq = row2[2]
                    # print('occurs at freq: ', freq)
                    full_df.loc[idx, 'frequency'] = freq
                    # print('\t matched ', row2[0], row2[1], "at freq: ", freq)
                # else: print('mag does not match, skipping')
            # else: print(row2[0], row2[1], "has no match... freq = 0")
    print('\n frequency df')
    print(full_df)

    # restructure databy direction bin
    frames3 = []
    for sub_dir in dir_dict:
        # print(sub_dir)
        vars()['grp_'+sub_dir] = full_df.loc[full_df[var_dir+'2'] == sub_dir]
        vars()['grp_'+sub_dir] = vars()['grp_'+sub_dir].sort_values([var_mag+'2'], ascending=(True))
        frames3.append(vars()['grp_'+sub_dir])
        
    reorder3 = pd.concat(frames3)
    print('\n reordered frequency df for plotting')
    print(reorder3)
    global rose_data  
    rose_data = reorder3.copy()
    
#%% plot rose on real data

plotly_rose(df_vars)

fig = px.bar_polar(rose_data, r="frequency", theta=var_dir+'2', color=var_mag+'2',
                   color_discrete_sequence= px.colors.sequential.Plasma_r,
                   # template="plotly_dark",
                   )
pio.renderers.default = 'browser'
fig.update_layout(font=dict(size=20))
fig.show()

# fig.write_image("Z:/your_save_path/figure.png", scale = 5)

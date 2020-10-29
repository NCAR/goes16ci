#!/usr/bin/env python

import sys
import matplotlib
import matplotlib.pylab as plt
import matplotlib.colors as colors
import numpy as np
import xarray as xr
import numpy.ma as ma
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.animation as animation
#access inputted files
#parse arguments
"""
Script to create animated visualizations for comparison of GOES16 prediction patches and real GLM data

Example Usage:
./prediction_vs_actual_visual.py "/glade/u/home/gwallach/goes16ci/scripts/lightning_20190601T18000_out.nc" "/glade/scratch/bpetzke/glm_grid_s20190601T180000_e20190601T230000.nc" > visual.log
"""

Pred = xr.open_dataset(sys.argv[1]) #Prediction Dataset
Actual_glm = xr.open_dataset(sys.argv[2]) #Actual Lightning Dataset
print(Pred)
print(Actual_glm)
#get data from datasets
light_prob = Pred.data_vars['lightning_prob']
actual_light = Actual_glm.data_vars['lightning_counts']
lons = Pred.data_vars['lon']
lats = Pred.data_vars['lat']
#clean up the actual dataset 
actual_light.values = actual_light.values.astype('float64')
actual_light.values[actual_light.values <= 0.0] = np.nan

#plot data and create animation
fig = plt.figure(figsize=(18,12))
ax = plt.axes(projection=ccrs.Miller())
states = cfeature.STATES.with_scale('10m')
ax.add_feature(states, edgecolor='black')
pred_bounds = np.array([0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
norm = colors.BoundaryNorm(boundaries=pred_bounds,ncolors=200,clip=True)
Pred = ax.pcolormesh(lons,lats,light_prob[0,:,:],transform=ccrs.Miller(),norm=norm,cmap='Reds')
Actual = ax.pcolormesh(lons,lats,actual_light[0,:,:],transform=ccrs.Miller(),cmap='Blues')
plt.colorbar(Actual,ax=ax,label='Actual Lightning Strikes')
plt.colorbar(Pred,ax=ax,label='Probability of Lightning Strikes')

def animation_frame(i):
    Pred = ax.pcolormesh(lons,lats,light_prob[i,:,:],transform=ccrs.Miller(),norm=norm,cmap='Reds',vmin=0.1,vmax=1,)
    Actual = ax.pcolormesh(lons,lats,actual_light[i,:,:],transform=ccrs.Miller(),cmap='Blues')
    #change the title to display the timestep for every time
    time = str(actual_light[i,:,:].time.coords)
    time = time.partition("[ns] ")[2]
    plt.title("Predicted vs Actual Lightning Strikes for (%s)" % time,fontsize='22')
    return Actual, Pred

def init():
    Pred = ax.pcolormesh(lons,lats,light_prob[0,:,:],transform=ccrs.Miller(),norm=norm,cmap='Reds')
    Actual = ax.pcolormesh(lons,lats,actual_light[0,:,:],transform=ccrs.Miller(),cmap='Blues')
    return Actual, Pred


start_time = str(actual_light[0,:,:].time.coords)
start_time = start_time.partition("[ns] ")[2]
end_time = str(actual_light[-1,:,:].time.coords)
end_time = end_time.partition("[ns] ")[2]
anim = animation.FuncAnimation(fig,animation_frame, init_func=init,frames=len(light_prob),interval=10, blit = True)
Writer = animation.PillowWriter()
anim.save('PredvsActual{}-{}.gif'.format(start_time,end_time),writer=Writer,dpi=400)
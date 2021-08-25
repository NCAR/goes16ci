#!/usr/bin/env python

import argparse
import pandas as pd
import netCDF4
import numpy as np
import torch
from tensorflow.keras.models import load_model
from goes16ci.models import MinMaxScaler2D

"""
Script to perform inference of a patchfile using a trained Tensorflow NN model

Example Usage:
#Midwest
./patch_inference.py -v "/glade/scratch/gwallach/goes16/goes16ci_model_cpu2020-06-10 16:14:15.932711.h5" /glade/p/ral/wsap/petzke/lightning/abi/patches/abi_patches_20190601T180000.nc /glade/p/ral/wsap/petzke/lightning/glm/grid/glm_grid_s20190601T180000_e20190601T230000.nc lightning_20190601T18000_out.nc
"""


def infer(modelfile,patchfile,glmtemplate,outfile,verbose=False):
    #for torch models uncomment next line
    model = torch.load('../best.pt')
    #for tensorflow models uncomment next line
    #model = load_model(modelfile)
    if verbose:
        model.summary()

    print("Loading data",flush=True)
    nc = netCDF4.Dataset(patchfile)
    abi_data = nc['abi'][:]
    time_data = nc['time'][:]
    lon_data = nc['lon'][:]
    lat_data = nc['lat'][:]

    patch_dim_size = nc.dimensions['patch'].size
    patch_y_dim_size = nc.dimensions['y'].size
    patch_x_dim_size = nc.dimensions['x'].size    
    nc.close()

    abi_data = np.swapaxes(abi_data,1,3)
    abi_data = np.swapaxes(abi_data,1,2)
    print("Not Scaled = ",abi_data.shape)
    #rescale ABI data to pass into Model
    scaler = MinMaxScaler2D()
    scaler.scale_values = pd.read_csv("../scale_values.csv")
    abi_data_scaled = 1.0 - scaler.transform(abi_data)
    print("Scaled = ",abi_data_scaled.shape)
    

    if verbose:
        print("abi_data.shape = %s, abi_data = %s" % (abi_data.shape,abi_data[[0,-1]]),flush=True)

    print("Performing inference",flush=True)
    #uncomment next line for tensorflow model
    #y_hat = model.predict(abi_data_scaled)
    #uncomment next line for pytorch model
    y_hat = model.predict(abi_data_scaled)
    print("Reading output template netcdf")
    nc = netCDF4.Dataset(glmtemplate)
    y_dim_size = nc.dimensions['y'].size
    x_dim_size = nc.dimensions['x'].size
    time_dim_size = nc.dimensions['time'].size
    nc.close()

    y_hat = y_hat.reshape((time_dim_size,y_dim_size,x_dim_size))
    if verbose:
        print("y_hat.shape (after reshape) = %s, y_hat = %s" % (y_hat.shape,y_hat),flush=True)

    print("Creating output file")
    ncout = netCDF4.Dataset(outfile, 'w', format='NETCDF4')

    ncout.createDimension('y', y_dim_size)
    ncout.createDimension('x', x_dim_size)
    ncout.createDimension('time', time_dim_size)

    mid_y = patch_y_dim_size//2
    mid_x = patch_x_dim_size//2

    lon_var = ncout.createVariable('lon', np.float64, ('y','x',))
    lon_var.fill_value = np.nan
    lon_var[:] = lon_data[:patch_dim_size//time_dim_size,mid_y,mid_x].reshape((y_dim_size,x_dim_size))

    lat_var = ncout.createVariable('lat', np.float64, ('y','x',))
    lat_var.fill_value = np.nan
    lat_var[:] = lat_data[:patch_dim_size//time_dim_size,mid_y,mid_x].reshape((y_dim_size,x_dim_size))

    time_var = ncout.createVariable('time', np.int64, ('time',))
    time_var.units = 'minutes since 2019-06-01 18:20:00'
    time_var.calendar = 'proleptic_gregorian'
    time_var[:] = np.unique(time_data)

    light_prob = ncout.createVariable('lightning_prob', np.float64, ('time','y','x',))
    light_prob.fill_value = np.nan
    light_prob[:] = y_hat

    ncout.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("modelfile",type=str,help="HDF5 model file")
    parser.add_argument("patchfile",type=str,help="Input nc patch file")
    parser.add_argument("glmtemplate",type=str,help="Output GLM template. Should use the same dimensions used to create patch file (i.e. cover the same date range length and use same lead_time/grid_freq).")
    parser.add_argument("outfile",type=str,help="Output nc file")
    parser.add_argument("-v","--verbose",dest="verbose",help="Provide extra debugging information",action="store_true")
    args = parser.parse_args()
    infer(args.modelfile,args.patchfile,args.glmtemplate,args.outfile,args.verbose)


if __name__ == '__main__':
    main()

#!/usr/bin/env python

import argparse

import netCDF4
import numpy as np
from tensorflow.keras.models import load_model

"""
Script to perform inference of a patchfile using a trained Tensorflow NN model

Example Usage:
#Random patches
./patch_inference.py "/glade/scratch/gwallach/goes16/goes16ci_model_cpu2020-06-10 16:14:15.932711.h5" "/glade/scratch/gwallach/goes16_nc/ABI_patches_32/abi_patches_20190915T000000.nc" lightning_20190915T000000_out.nc
#Midwest
./patch_inference.py "/glade/scratch/gwallach/goes16/goes16ci_model_cpu2020-06-10 16:14:15.932711.h5" "/glade/p/ral/wsap/petzke/lightning/abi/patches/abi_patches_20190601T180000.nc" lightning_20190601T18000_out.nc
"""


def infer(modelfile,patchfile,outfile,verbose=False):
    model = load_model(modelfile)
    if verbose:
        model.summary()

    print("Loading data",flush=True)
    nc = netCDF4.Dataset(patchfile)
    abi_data = nc['abi'][:]
    time_data = nc['time'][:]
    lon_data = nc['lon'][:]
    lat_data = nc['lat'][:]
    patch_data = nc['patch'][:]

    patch_dim_size = nc.dimensions['patch'].size
    y_dim_size = nc.dimensions['y'].size
    x_dim_size = nc.dimensions['x'].size    
    nc.close()

    abi_data = np.swapaxes(abi_data,1,3)
    abi_data = np.swapaxes(abi_data,1,2)
    if verbose:
        print("abi_data.shape = %s, abi_data = %s" % (abi_data.shape,abi_data[[0,-1]]),flush=True)

    print("Performing inference",flush=True)
    y_hat = model.predict(abi_data)

    y_hat = y_hat.reshape((len(y_hat),))
    if verbose:
        print("y_hat.shape = %s, y_hat = %s" % (y_hat.shape,y_hat),flush=True)

    print("Creating output file")
    ncout = netCDF4.Dataset(outfile, 'w', format='NETCDF4')

    ncout.createDimension('patch', patch_dim_size)

    prob_light = ncout.createVariable('prob_lightning', np.dtype(np.float32).char, ('patch',))
    prob_light.fill_value = np.nan
    prob_light[:] = y_hat

    time_var = ncout.createVariable('time', np.dtype(np.int64).char, ('patch',))
    time_var.units = 'minutes since 2019-06-01 18:20:00'
    time_var.calendar = 'proleptic_gregorian'
    time_var[:] = time_data

    mid_y = y_dim_size//2
    mid_x = x_dim_size//2

    lon_var = ncout.createVariable('lon', np.dtype(np.float32).char, ('patch',))
    lon_var.fill_value = np.nan
    lon_var[:] = lon_data[:,mid_y,mid_x]

    lat_var = ncout.createVariable('lat', np.dtype(np.float32).char, ('patch',))
    lat_var.fill_value = np.nan
    lat_var[:] = lat_data[:,mid_y,mid_x]

    patch_var = ncout.createVariable('patch', np.dtype(np.int64).char, ('patch',))
    patch_var[:] = patch_data

    ncout.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("modelfile",type=str,help="HDF5 model file")
    parser.add_argument("patchfile",type=str,help="Input nc patch file")
    parser.add_argument("outfile",type=str,help="Output nc file")
    parser.add_argument("-v","--verbose",dest="verbose",help="Provide extra debugging information",action="store_true")
    args = parser.parse_args()
    infer(args.modelfile,args.patchfile,args.outfile,args.verbose)


if __name__ == '__main__':
    main()

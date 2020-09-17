#!/usr/bin/env python

import argparse
from datetime import datetime
import logging
import sys
import math
import os.path
import os

import yaml
import pandas as pd
import numpy as np
from haversine import haversine, Unit
from dask.distributed import LocalCluster, Client, as_completed, wait

from goes16ci.lightning import create_glm_grids
from goes16ci.imager import extract_all_abi_patches
from goes16ci.imager import extract_abi_patches

"""
Program creates patch files for input into CNN models for user specified bounds and date range.

Example Usage: 

CONUS
./build_patches.py -125.0 25 -66 50 2019-06-01T00:00:00 2019-06-01T23:59:59 goes16_build_patches_config_32.yml

MIDWEST
./build_patches.py -105 30 -85 50 2019-06-01T18:00:00 2019-06-01T23:00:00 goes16_build_patches_config_32.yml
"""

#Parallelization parameters
PARALLEL=False
N_WORKERS=12
THREADS_PER_WORKER=2
WORKER_MEM='32GB'

logging.basicConfig(format='%(asctime)s %(message)s',
                    level=logging.DEBUG,
                    datefmt="%Y-%m-%dT%H:%M:%S")


def valid_iso_date(ds):
    try:
        return datetime.strptime(ds, "%Y-%m-%dT%H:%M:%S")
    except ValueError:
        msg = "Not a valid ISO8601 yyyy-mm-ddThh:mm:ss date: '{0}'.".format(ds)
        raise argparse.ArgumentTypeError(msg)


def valid_lat(y):
    try:
        y = float(y)
    except ValueError:
        msg = "Not a valid latitude: '{0}'.".format(y)
        raise argparse.ArgumentTypeError(msg)

    if y < -90 or 90 < y:
        msg = "Not a valid latitude: '{0}'.".format(y)
        raise argparse.ArgumentTypeError(msg)
    return y


def valid_lon(x):
    try:
        x = float(x)
    except ValueError:
        msg = "Not a valid longitude: '{0}'.".format(x)
        raise argparse.ArgumentTypeError(msg)

    if x < -180 or 180 < x:
        msg = "Not a valid longitude: '{0}'.".format(x)
        raise argparse.ArgumentTypeError(msg)
    return x


def build_patches(min_lon, min_lat, max_lon, max_lat, begin_iso_dt, end_iso_dt, config_yaml):
    logging.info("Executing: %s(%s)" % ("build_patches", locals()))

    # Projection definition for Pyproj (PROJ4 keywords)
    grid_proj_params = {"proj":"lcc","lon_0":(min_lon+max_lon)/2,"lat_0":(min_lat+max_lat)/2,"lat_1":min_lat,"lat_2":max_lat}
    print(grid_proj_params)
    with open(config_yaml, "r") as config_file:
        config = yaml.load(config_file)

    glm_config = config["glm"]
    glm_path = glm_config["glm_path"]
    grid_path = glm_config["grid_path"]
    file_freq = glm_config["file_freq"]
    glm_file_dates = pd.DatetimeIndex(pd.date_range(start=begin_iso_dt, end=end_iso_dt, freq=file_freq))
    grid_freq = glm_config["grid_freq"]
    dx_km = glm_config["dx_km"]
    x_extent_km = int(math.ceil(haversine((min_lat,min_lon),(min_lat,max_lon),unit=Unit.KILOMETERS)))
    y_extent_km = int(math.ceil(haversine((min_lat,min_lon),(max_lat,min_lon),unit=Unit.KILOMETERS)))
    
    if not os.path.exists(grid_path):
        os.makedirs(grid_path)

    if PARALLEL:
       cluster = LocalCluster(n_workers=N_WORKERS, processes=True, threads_per_worker=THREADS_PER_WORKER, memory_limit=WORKER_MEM)
       client = Client(cluster)
       glm_jobs = []
       for date in glm_file_dates:
           logging.info("Processing: %s", date)
           glm_jobs.append(client.submit(create_glm_grids, glm_path, grid_path, date, min(end_iso_dt,date + pd.Timedelta(file_freq)), grid_freq, grid_proj_params, dx_km, x_extent_km, y_extent_km))
       for glm_job in as_completed(glm_jobs):
           res = glm_job.result()
           if glm_job.status == "error":
               traceback.format_tb(res[-1])
       del glm_jobs[:]
    else:
       for date in glm_file_dates:
           logging.info("Processing: %s", date)
           create_glm_grids(glm_path, grid_path, date, min(end_iso_dt,date + pd.Timedelta(file_freq)), grid_freq, grid_proj_params, dx_km, x_extent_km, y_extent_km)

    abi_config = config["abi"]
    abi_path = abi_config["abi_path"]
    patch_path = abi_config["patch_path"]
    glm_grid_path = abi_config["glm_grid_path"]
    bands = np.array(abi_config["bands"])
    file_freq = abi_config["file_freq"]
    lead_time = abi_config["lead_time"]
    patch_x_length_pixels = abi_config["patch_x_length_pixels"]
    patch_y_length_pixels = abi_config["patch_y_length_pixels"]
    time_range_minutes = abi_config["time_range_minutes"]
    bt = bool(abi_config["bt"])

    if not os.path.exists(patch_path):
        makedirs(patch_path)

    abi_file_dates = pd.DatetimeIndex(pd.date_range(start=begin_iso_dt, end=end_iso_dt, freq=file_freq))
    
    if PARALLEL:
        abi_jobs = []
        for date in abi_file_dates:
            abi_jobs.append(client.submit(extract_all_abi_patches, abi_path, patch_path, glm_grid_path, date, 
                                          min(end_iso_dt,date + pd.Timedelta(file_freq)), bands,
                                          lead_time, patch_x_length_pixels, patch_y_length_pixels,
                                          time_range_minutes=time_range_minutes, glm_file_freq=file_freq, bt=bt))
        # for abi_job in as_completed(abi_jobs):
        #     res = abi_job.result()
        #     if abi_job.status == "error":
        #         print(traceback.format_tb(res[-1]),flush=True)
        wait(abi_jobs)
        abi_results = client.gather(abi_jobs)
        del abi_jobs[:]
        client.close()
    else:
        for date in abi_file_dates:
            extract_all_abi_patches(abi_path, patch_path, glm_grid_path, date,
                                    min(end_iso_dt,date + pd.Timedelta(file_freq)), bands,
                                    lead_time, patch_x_length_pixels, patch_y_length_pixels,
                                    time_range_minutes=time_range_minutes, glm_file_freq=file_freq, bt=bt)


def main():
    parser = argparse.ArgumentParser(description='Creates patch data for input into CNN inference modules.')
    parser.add_argument('min_lon', type=valid_lon, help="Minimum longitude")
    parser.add_argument('min_lat', type=valid_lat, help="Minimum latitude")
    parser.add_argument('max_lon', type=valid_lon, help="Maximum longitude")
    parser.add_argument('max_lat', type=valid_lat, help="Maximum latitude")
    parser.add_argument('begin_iso_dt', type=valid_iso_date, help="ISO8601 date format yyyy-mm-ddThh:mm:ss")
    parser.add_argument('end_iso_dt', type=valid_iso_date, help="ISO8601 date format yyyy-mm-ddThh:mm:ss")
    parser.add_argument("config_yaml", help="Config yaml file")
    args = parser.parse_args()
    
    logging.info("Starting %s" % sys.argv[0])
    logging.info("Processing patch files")
    
    build_patches(**vars(args))

    logging.info("Finishing %s" % sys.argv[0])


if __name__ == '__main__':
    main()

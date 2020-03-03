import xarray as xr
import numpy as np
from glob import glob
from os.path import join
import pandas as pd
from dask.distributed import LocalCluster, Client, wait
import logging

def load_single_data_file(filename, image_variable="abi", count_variable="flash_counts", time_variable="time"):
    ds = xr.open_dataset(filename)
    imagery = ds.variables[image_variable].values
    nan_indices = np.unique(np.where(np.isnan(imagery))[0])
    all_indices = np.arange(imagery.shape[0])
    valid_indices = all_indices[np.isin(all_indices, nan_indices, assume_unique=True, invert=True)]
    good_imagery = imagery[valid_indices]
    counts = ds.variables[count_variable][valid_indices].values
    time = ds.variables[time_variable][valid_indices].values
    ds.close()
    return good_imagery, counts, time


def load_data_serial(data_path, image_variable="abi", count_variable="flash_counts", time_variable="time",
                     start_date=None, end_date=None):
    data_files = sorted(glob(join(data_path, "*.nc")))
    images_list = []
    counts_list = []
    time_list = []
    if start_date is not None and end_date is not None:
        start_date = pd.Timestamp(start_date)
        end_date = pd.Timestamp(end_date)
    else:
        start_date = None
        end_date = None
    for data_file in data_files:
        if start_date is not None:
            file_date = pd.Timestamp(data_file.split("/")[-1][:-3].split("_")[-1])
            if (file_date < start_date) or (file_date > end_date):
                continue
        logging.info(data_file)
        images, counts, time = load_single_data_file(data_file, image_variable=image_variable,
                                                     count_variable=count_variable, time_variable=time_variable)
        images_list.append(images)
        counts_list.append(counts)
        time_list.append(time)
    all_images = np.concatenate(images_list)
    all_images = np.moveaxis(all_images, 1, -1)
    all_counts = np.concatenate(counts_list)
    all_time = pd.DatetimeIndex(np.concatenate(time_list))
    return all_images, all_counts, all_time


def load_data_parallel(data_path, num_processes,
                       image_variable="abi", count_variable="flash_counts", time_variable="time"):
    cluster = LocalCluster(n_workers=num_processes, threads_per_worker=1)
    client = Client(cluster)
    data_files = sorted(glob(join(data_path, "*.nc")))
    data_jobs = []
    for data_file in data_files:
        data_jobs.append(client.submit(load_single_data_file, data_file,
                                       image_variable=image_variable,
                                       count_variable=count_variable,
                                       time_variable=time_variable))
    wait(data_jobs)
    data_results = client.gather(data_jobs)
    all_images = np.concatenate([d[0] for d in data_results])
    all_counts = np.concatenate([d[1] for d in data_results])
    all_time = pd.DatetimeIndex(np.concatenate([d[2] for d in data_results]))
    client.close()
    cluster.close()
    del client
    del cluster
    return all_images, all_counts, all_time


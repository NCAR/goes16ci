import xarray as xr
import numpy as np
from glob import glob
from os.path import join
import pandas as pd
from dask.distributed import LocalCluster, Client, wait
import logging
import s3fs as s3
import os

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
    if len(data_files) == 0:
        logging.error("No data files available in the data directory. Please run\npython download_data.py\n on an internet-connected node to retrieve the data.")
        raise FileNotFoundError("No data files found in the data directory.")
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

def split_data(train_start, train_end, val_start, val_end, test_start, test_end, all_data, all_counts, all_time):
    train_indices = np.where((all_time >= train_start) & (all_time <=train_end))[0]
    val_indices  = np.where((all_time >= val_start) & (all_time <=val_end))[0] 
    test_indices = np.where((all_time >= test_start) & (all_time <=test_end))[0]
    train_data = all_data[train_indices].astype("float32")
    val_data = all_data[val_indices].astype("float32")
    test_data = all_data[test_indices].astype("float32")
    train_counts = np.where(all_counts[train_indices] > 0, 1, 0).astype('float32')
    val_counts = np.where(all_counts[val_indices] > 0, 1, 0).astype('float32')
    test_counts = np.where(all_counts[test_indices] > 0, 1, 0).astype('float32')
    data_subsets = {}
    counts_subsets = {}
    data_subsets['train'] = train_data
    data_subsets['val'] = val_data
    data_subsets['test'] = test_data
    counts_subsets['train'] = train_counts
    counts_subsets['val'] = val_counts
    counts_subsets['test'] = test_counts
    return data_subsets, counts_subsets

def download_data(start_date,end_date,instrument,sector,outpath):   
    fs = s3.S3FileSystem(anon=True)
    ins_sec = instrument + '-' + sector
    try:
        os.stat(outpath + ins_sec)
    except:
        os.mkdir(outpath + ins_sec)
    year = start_date.split('-')[0]
    date = pd.to_datetime(start_date, format='%Y-%m-%d')
    new_year_day = pd.Timestamp(year=date.year, month=1, day=1)
    start_day = (date - new_year_day).days + 1
    
    date = pd.to_datetime(end_date, format='%Y-%m-%d')
    new_year_day = pd.Timestamp(year=date.year, month=1, day=1)
    end_day = (date - new_year_day).days + 1
    
    for day in range(start_day,end_day + 1):
        day = str(day)
        if len(day) == 1:
            day = '00' + day
        if len(day) == 2:
            day = '0' + day
        #print(day)
        path = ins_sec + '/' + year + day
        #print(path)
        try:
            os.stat(outpath + path)
        except:
            os.mkdir(outpath + path)
        for hour in range(24):
            hour = str(hour)
            if len(hour) == 1:
                hour = '0' + hour
            files = fs.ls('s3://noaa-goes16/' + ins_sec + '/' + year + '/'+ day +'/'+ hour)
            for file in files:
                fs.get(file, outpath + '/' + path + '/' + file.split('/')[-1])
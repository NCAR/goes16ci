import xarray as xr
import numpy as np
import pandas as pd
from pyproj import Proj
import glob as glob


from os.path import join, exists
from os import makedirs
from datetime import datetime
import time
from dask import compute
import dask.array as da
import logging
import pathlib
import os as os
import math
# from haversine import haversine, Unit

def load_nldn_data(path, start_date, end_date, freq="1S",
                  columns=('Date', 'Time', 'Lat', 'Lon', 'PeakCurrent' ,'Classification')):
    """
    Read a concurrent set of nldn files and extract the flash location data into a :class:`pandas.DataFrame`.
    Args:
        path (str): Path to nldn directory
        start_date (str or :class:`pandas.Timestamp`):
        end_date (str or :class:`pandas.Timestamp`):
        freq (str): pandas time frequency code. Default is "20S" or 20 seconds, which is the approximate frequency
            of nldn data updates
        columns (list or tuple): variables to be extracted from the NLDN file. The longitude and latitude are most
            important
    Returns:
        :class:`pandas.DataFrame`: all of the flash events during the specified time period.
    """

    all_times = pd.DatetimeIndex(pd.date_range(start=start_date, end=end_date,  freq="20s"))[0:-1]
#     print(start_date, end_date)
    print(all_times[0], all_times[-1])
    print("Within Load NLDN Data Function")
    all_dates = np.unique(all_times.date)
    all_date_strings = [date.strftime("%Y-%m-%d") for date in all_dates]
    all_flashes = []

    for date_str in all_date_strings:
        #print("Within for loop for date: ", date_str)
        comb_str = date_str + ".parquet.gzip"
        nldn_date_files= os.path.join(path, comb_str)
        file_exists = exists(nldn_date_files)
        if(not file_exists):
            print(f"No NLDN Files Found for Date {date_str}")
            continue
        else:
            nldn_ds = pd.read_parquet(nldn_date_files)
            if nldn_ds[columns[0]].shape[0] > 0:
                nldn_ds['Timestamp'] = pd.to_datetime(nldn_ds['Date'] + " " + nldn_ds['Time'])
                curr_lightning = nldn_ds.loc[(nldn_ds['Timestamp'] >= all_times[0]) & 
                                               (nldn_ds['Timestamp'] <= all_times[-1])]
                if (curr_lightning.empty):
                    continue
                else:
                    all_flashes.append(curr_lightning)
                    
    if len(all_flashes) == 0:
        return None
    else:
        combined_flashes = pd.concat(all_flashes)
        return combined_flashes

class NLDNGrid(object):
    """
    A projected uniform grid for discretizing the flashes into a binary occurrence grid.
    Attributes:
        proj_params (dict): map projection parameters for pyproj. Should contain lat_0 and lon_0 to specify grid center.
        dx_km (float): The grid spacing in kilometers
        x_extent_km (float): The west-east length of the grid domain in kilometers
        y_extent_km (float): The south-north length of the grid domain in kilometers
        nldn_proj (`pyproj.Proj`): Projection object
        x_points (:class:`numpy.ndarray`): The x coordinate for the center of each grid box.
        y_points (:class:`numpy.ndarray`): The y coordinate for the center of each grid box.
        x_grid (:class:`numpy.ndarray` [y, x]): The x values for each grid point.
        y_grid (:class:`numpy.ndarray` [y, x]): The y values for each grid point.
        lon_grid (:class:`numpy.ndarray` [y, x]): The longitudes for each grid point.
        lat_grid (:class:`numpy.ndarray` [y, x]): The longitudes for each grid point.
    """
    def __init__(self, proj_params, dx_km, x_extent_km, y_extent_km):
        self.proj_params = proj_params
        self.dx_km = dx_km
        self.x_extent_km = x_extent_km
        self.y_extent_km = y_extent_km
        self.nldn_proj = Proj(**proj_params)
        self.x_points = np.arange(-x_extent_km / 2, x_extent_km / 2 + dx_km, dx_km)
        self.y_points = np.arange(-y_extent_km / 2, y_extent_km / 2 + dx_km, dx_km)
        print(self.x_points)
        print(self.y_points)
        self.x_grid, self.y_grid = np.meshgrid(self.x_points, self.y_points)
        self.lon_grid, self.lat_grid = self.nldn_proj(self.x_grid * 1000, self.y_grid * 1000, inverse=True)

    def grid_NLDN_data(self, flashes):
        """
        Aggregate the point flashes into a grid of flash counts occurring within each grid box.
        Args:
            flashes (:class:`pandas.DataFrame`): Contains the longitudes and latitudes of each flash
        Returns:
            :class:`numpy.ndarray` [y, x]: The number of flashes occurring at each grid point.
        """
        flash_x, flash_y = self.nldn_proj(flashes["Lon"].values, flashes["Lat"].values)
        print(flash_x)
        flash_x /= 1000
        flash_y /= 1000
        valid_flashes = np.where((flash_x >= self.x_points.min() - self.dx_km / 2) &
                                 (flash_x <= self.x_points.max() + self.dx_km / 2) &
                                 (flash_y >= self.y_points.min() - self.dx_km / 2) &
                                 (flash_y <= self.y_points.max() + self.dx_km / 2))[0]
        #we have chosen to uncomment the parallelized code
        if valid_flashes.size > 0:
            if PARALLEL:
                x_grid_flat = da.from_array(self.x_grid.reshape((self.x_grid.size, 1)),chunks=512)
                y_grid_flat = da.from_array(self.y_grid.reshape((self.x_grid.size, 1)),chunks=512)
                flash_x_flat = da.from_array(flash_x[valid_flashes].reshape(1, valid_flashes.size),chunks=512)
                flash_y_flat = da.from_array(flash_y[valid_flashes].reshape(1, valid_flashes.size),chunks=512)
                x_dist = da.fabs(x_grid_flat - flash_x_flat)
                y_dist = da.fabs(y_grid_flat - flash_y_flat)
                flash_grid_counts = da.sum((x_dist <= self.dx_km / 2) & (y_dist <= self.dx_km / 2), axis=1)
                flash_grid = flash_grid_counts.reshape(self.lon_grid.shape).astype(np.int32).compute()
            else:
                x_grid_flat = self.x_grid.reshape((self.x_grid.size, 1))
                y_grid_flat = self.y_grid.reshape((self.x_grid.size, 1))
                flash_x_flat = flash_x[valid_flashes].reshape(1, valid_flashes.size)
                flash_y_flat = flash_y[valid_flashes].reshape(1, valid_flashes.size)
                x_dist = np.abs(x_grid_flat - flash_x_flat)/
                y_dist = np.abs(y_grid_flat - flash_y_flat)
                flash_grid_counts = np.sum((x_dist <= self.dx_km / 2) & (y_dist <= self.dx_km / 2), axis=1)
                flash_grid = flash_grid_counts.reshape(self.lon_grid.shape).astype(np.int32)
        else:
            flash_grid = np.zeros(self.lon_grid.shape, dtype=np.int32)
        return flash_grid



    def create_NLDN_grids(nldn_path, out_path, start_date, end_date, out_freq,
                     grid_proj_params, dx_km, x_extent_km, y_extent_km, return_grid=False):
    """
    For a given time range, load NLDN data and grid it at regular time intervals. After gridding the data, save it
    to a netCDF file.
    other useful parameters: file_freq: "1D" # How often to write separate netCDF files
        grid_freq: "20Min" # Aggregation period for each gridded time step
        dx_km: # Horizontal grid spacing in km
        
    Args:
        NLDN_path: 
        out_path:
        start_date: specified start date
        end_date:   specified end date
        out_freq: 20Min
        grid_proj_params: 
        dx_km: "16.0"
        x_extent_km:
        y_extent_km:
        return_grid:
    
    Returns: a NLDN Grid
    
    
    """
    print(start_date, end_date, flush=True)
    out_dates = pd.DatetimeIndex(pd.date_range(start=start_date, end=end_date, freq=out_freq))
    grid = NLDNGrid(grid_proj_params, dx_km, x_extent_km, y_extent_km)
    flash_count_grid = xr.DataArray(np.zeros((out_dates.size - 1, grid.y_points.size, grid.x_points.size),
                                             dtype=np.int32),
                                    coords={"y": grid.y_points, "x": grid.x_points, "lon":(("y", "x"), grid.lon_grid),
                                            "lat": (("y", "x"), grid.lat_grid),
                                            "time": out_dates[1:]}, dims=("time", "y", "x"),
                                    name="lightning_counts")
    print(out_dates, flush=True)
    for o in range(1, out_dates.shape[0]):
        period_start = out_dates[o - 1]
        period_end = out_dates[o]
        start_t = datetime.now()
        print(period_start, period_end)
        period_flashes = load_nldn_data(nldn_path, period_start, period_end)
        print("load_NLDN_data runtime: %d" % (datetime.now()- start_t).total_seconds())
        if period_flashes is not None:
            print("Registered Period Flashes", period_flashes)
            start_t = datetime.now()
            flash_count_grid[o - 1] = grid.grid_NLDN_data(period_flashes)
            print("grid_nldn_data runtime: %d" % (datetime.now()- start_t).total_seconds())
        else:
            print("Period Flashes is None")
        print(out_dates[o], flash_count_grid[o - 1].values.max(), flash_count_grid[o - 1].values.sum(), flush=True)
        del period_flashes
    flash_count_grid.attrs.update(grid_proj_params)
    # print("Flash Count Check",flash_count_grid)
    out_file = join(out_path,
                    f"nldn_grid_s{start_date.strftime('%Y%m%dT%H%M%S')}_e{end_date.strftime('%Y%m%dT%H%M%S')}.nc")
    if not exists(out_path):
        makedirs(out_path)
    flash_count_grid.to_netcdf(out_file, encoding={"lightning_counts": {"zlib": True}})
    if return_grid:
        return flash_count_grid
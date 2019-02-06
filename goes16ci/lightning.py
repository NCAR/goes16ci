import xarray as xr
import numpy as np
import pandas as pd
from pyproj import Proj
from glob import glob
from os.path import join
from datetime import datetime


def load_glm_data(path, start_date, end_date, freq="20S",
                  columns=("flash_lon", "flash_lat", "flash_energy")):
    all_times = pd.DatetimeIndex(start=start_date, end=end_date, freq=freq)
    all_dates = np.unique(all_times.date)
    all_date_strings = [date.strftime("%Y%m%d") for date in all_dates]
    all_flashes = []
    for date_str in all_date_strings:
        glm_date_files = sorted(glob(join(path, date_str, "*.nc")))
        for glm_date_file in glm_date_files:
            file_start_date = pd.Timestamp(datetime.strptime(glm_date_file[:-3].split("/")[-1].split("_")[4][1:-1],
                                                             "%Y%j%H%M%S"))
            if all_times[0] <= file_start_date <= all_times[-1]:
                glm_ds = xr.open_dataset(glm_date_file)
                all_flashes.append(glm_ds[columns].to_dataset())
                glm_ds.close()
                del glm_ds
    combined_flashes = pd.concat(all_flashes)
    return combined_flashes


class GLMGrid(object):
    def __init__(self, proj_params, dx_km, x_extent_km, y_extent_km):
        self.proj_params = proj_params
        self.dx_km = dx_km
        self.x_extent_km = x_extent_km
        self.y_extent_km = y_extent_km
        self.glm_proj = Proj(**proj_params)
        self.x_points = np.arange(-x_extent_km / 2, x_extent_km / 2 + dx_km, dx_km)
        self.y_points = np.arange(-y_extent_km / 2, y_extent_km / 2 + dx_km, dx_km)
        self.x_grid, self.y_grid = np.meshgrid(self.x_points, self.y_points)
        self.lon_grid, self.lat_grid = self.glm_proj(self.x_grid * 1000, self.y_grid * 1000, inverse=True)

    def grid_glm_data(self, flashes):
        flash_x, flash_y = self.glm_proj(flashes["flash_lon"].values, flashes["flash_lat"].values)
        flash_x /= 1000
        flash_y /= 1000
        valid_flashes = np.where((flash_x >= self.x_points.min()) & (flash_x <= self.x_points.max()) &
                                (flash_y >= self.y_points.min()) & (flash_y) <= self.y_points.max())[0]
        if valid_flashes.size > 0:
            x_grid_flat = self.x_grid.reshape((self.x_grid.size, 1))
            y_grid_flat = self.y_grid.reshape((self.y_grid.size, 1))
            flash_x_flat = flash_x[valid_flashes].reshape(1, valid_flashes.size)
            flash_y_flat = flash_y[valid_flashes].reshape(1, valid_flashes.size)
            x_dist = np.abs(x_grid_flat - flash_x_flat)
            y_dist = np.abs(y_grid_flat - flash_y_flat)
            flash_grid_counts = np.sum((x_dist <= self.dx_km / 2) & (y_dist <= self.dx_km / 2), axis=1)
            flash_grid = np.where(flash_grid_counts.reshape(valid_flashes.shape) > 0, 1, 0).astype(np.int32)
        else:
            flash_grid = np.zeros(self.lon_grid.shape, dtype=np.int32)
        return flash_grid

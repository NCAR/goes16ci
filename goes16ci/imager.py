import pandas as pd
import numpy as np
from glob import glob
from datetime import datetime
from pyproj import Proj, transform
from os.path import join, exists
from os import makedirs
from scipy.interpolate import RectBivariateSpline
from sklearn.metrics import pairwise_distances_argmin
import xarray as xr
import time as cpytime


class GOES16ABI(object):
    """
    Handles data I/O and map projections for GOES-16 Advanced Baseline Imager data.

    Attributes:
        date (:class:`pandas.Timestamp`): Date of image slices
        bands (:class:`numpy.ndarray`): GOES-16 hyperspectral bands to load
        path (str): Path to top level of GOES-16 ABI directory.
        time_range_minutes (int): interval in number of minutes to search for file that matches input time
        goes16_ds (`dict` of :class:`xarray.Dataset` objects): Datasets for each channel

    """
    def __init__(self, date, bands, path, time_range_minutes=5):
        self.date = pd.Timestamp(date)
        self.bands = np.array(bands, dtype=np.int32)
        self.path = path
        self.time_range_minutes = time_range_minutes
        self.goes16_ds = dict()
        self.channel_files = []
        for band in bands:
            self.channel_files.append(self.goes16_abi_filename(band))
            self.goes16_ds[band] = xr.open_dataset(self.channel_files[-1])
        self.proj = self.goes16_projection()
        self.x = None
        self.y = None
        self.x_g = None
        self.y_g = None
        self.lon = None
        self.lat = None
        self.sat_coordinates()
        self.lon_lat_coords()

    @staticmethod
    def abi_file_dates(files, file_date='e'):
        """
        Extract the file creation dates from a list of GOES-16 files.
        Date format: Year (%Y), Day of Year (%j), Hour (%H), Minute (%M), Second (%s), Tenth of a second
        See `AWS <https://docs.opendata.aws/noaa-goes16/cics-readme.html>`_ for more details

        Args:
            files (list): list of GOES-16 filenames.
            file_date (str): Date in filename to extract. Valid options are
                's' (start), 'e' (end), and 'c' (creation, default).
        Returns:
            :class:`pandas.DatetimeIndex`: Dates for each file
        """
        if file_date not in ['c', 's', 'e']:
            file_date = 'c'
        date_index = {"c": -1, "s": -3, "e": -2}
        channel_dates = pd.DatetimeIndex(
            [datetime.strptime(c_file[:-3].split("/")[-1].split("_")[date_index[file_date]][1:-1],
                               "%Y%j%H%M%S") for c_file in files])
        return channel_dates

    def goes16_abi_filename(self, channel):
        """
        Given a path to a dataset of GOES-16 files, find the netCDF file that matches the expected
        date and channel, or band number.

        The GOES-16 path should point to a directory containing a series of directories named by
        valid date in %Y%m%d format. Each directory should contain Level 1 CONUS sector files.


        Args:
            channel (int): GOES-16 ABI `channel <https://www.goes-r.gov/mission/ABI-bands-quick-info.html>`_.
        Returns:
            str: full path to requested GOES-16 file
        """
        pd_date = pd.Timestamp(self.date)
        #print("pd_date ",pd_date)
        channel_files = np.array(sorted(glob(join(self.path, pd_date.strftime("%Y%m%d"),
                                         f"OR_ABI-L1b-RadC-M*C{channel:02d}_G16_*.nc"))))
        #print("channel_files",channel_files)
        channel_dates = self.abi_file_dates(channel_files)
        #print("channel_dates",channel_dates)
        date_diffs = np.abs(channel_dates - pd_date)
        #print("Date_diffs",date_diffs)
        file_index = np.where(date_diffs <= pd.Timedelta(minutes=self.time_range_minutes))[0]
        #print("File_index",file_index)
        if len(file_index) == 0:
            raise FileNotFoundError('No GOES-16 files within {0:d} minutes of '.format(self.time_range_minutes) + pd_date.strftime("%Y-%m-%d %H:%M:%S" + ". Nearest file is within {0}".format(date_diffs.total_seconds().values.min() / 60)))
        else:
            filename = channel_files[np.argmin(date_diffs)]
        return filename

    def goes16_projection(self):
        """
        Create a Pyproj projection object with the projection information from a GOES-16 file.
        The geostationary map projection is described in the
        `PROJ <https://proj4.org/operations/projections/geos.html>`_ documentation.

        """
        goes16_ds = self.goes16_ds[self.bands.min()]
        proj_dict = dict(proj="geos",
                         h=goes16_ds["goes_imager_projection"].attrs["perspective_point_height"],
                         lon_0=goes16_ds["goes_imager_projection"].attrs["longitude_of_projection_origin"],
                         sweep=goes16_ds["goes_imager_projection"].attrs["sweep_angle_axis"])
        return Proj(projparams=proj_dict)

    def sat_coordinates(self):
        """
        Calculate the geostationary projection x and y coordinates in m for each
        pixel in the image.
        """
        goes16_ds = self.goes16_ds[self.bands.min()]
        sat_height = goes16_ds["goes_imager_projection"].attrs["perspective_point_height"]
        self.x = goes16_ds["x"].values * sat_height
        self.y = goes16_ds["y"].values * sat_height
        self.x_g, self.y_g = np.meshgrid(self.x, self.y)

    def lon_lat_coords(self):
        """
        Calculate longitude and latitude coordinates for each point in the GOES-16
        image.
        """
        self.lon, self.lat = self.proj(self.x_g, self.y_g, inverse=True)
        self.lon[self.lon > 1e10] = np.nan
        self.lat[self.lat > 1e10] = np.nan

    def extract_image_patch(self, center_lon, center_lat, x_size_pixels, y_size_pixels, bt=True):
        """
        Extract a subset of a satellite image around a given location.

        Args:
            center_lon (float): longitude of the center pixel of the image
            center_lat (float): latitude of the center pixel of the image
            x_size_pixels (int): number of pixels in the west-east direction
            y_size_pixels (int): number of pixels in the south-north direction
            bt (bool): Convert to brightness temperature during extraction
        Returns:

        """
        center_x, center_y = self.proj(center_lon, center_lat)
        center_row = np.argmin(np.abs(self.y - center_y))
        center_col = np.argmin(np.abs(self.x - center_x))
        row_slice = slice(int(center_row - y_size_pixels // 2), int(center_row + y_size_pixels // 2))
        col_slice = slice(int(center_col - x_size_pixels // 2), int(center_col + x_size_pixels // 2))
        patch = np.zeros((1, self.bands.size, y_size_pixels, x_size_pixels), dtype=np.float32)
        
        for b, band in enumerate(self.bands):
            if bt:
                patch[0, b, :, :] = (self.goes16_ds[band]["planck_fk2"].values /
                                     np.log(self.goes16_ds[band]["planck_fk1"].values /
                                            self.goes16_ds[band]["Rad"][row_slice, col_slice].values + 1) -
                                     self.goes16_ds[band]["planck_bc1"].values) / self.goes16_ds[band]["planck_bc2"].values
            else:
                patch[0, b, :, :] = self.goes16_ds[band]["Rad"][row_slice, col_slice].values
        lons = self.lon[row_slice, col_slice]
        lats = self.lat[row_slice, col_slice]
        return patch, lons, lats

    def calc_nearest_rows_cols(self,lons,lats):
        wgs84 = Proj(init='epsg:4326')
        center_xs, center_ys = transform(wgs84,self.proj,lons,lats)
        center_xs, center_ys = center_xs.reshape(-1,1),center_ys.reshape(-1,1)
        center_rows = pairwise_distances_argmin(self.y.reshape(-1,1),center_ys,axis=0,metric='l1')
        center_cols = pairwise_distances_argmin(self.x.reshape(-1,1),center_xs,axis=0,metric='l1')
        return center_rows,center_cols


    def extract_all_image_patchs(self, center_lons, center_lats, x_size_pixels, y_size_pixels, bt=True):
        center_rows,center_cols = self.calc_nearest_rows_cols(center_lons.ravel(),center_lats.ravel())
        assert center_lons.size == center_lats.size == center_rows.size == center_cols.size
        patches = np.ones((center_lons.size, self.bands.size, y_size_pixels, x_size_pixels), dtype=np.float32)*np.nan
        lons = np.ones((center_lons.size,x_size_pixels,y_size_pixels),dtype=np.float32)*np.nan
        lats = np.ones((center_lons.size,x_size_pixels,y_size_pixels),dtype=np.float32)*np.nan
        half_x_size_pixels = x_size_pixels // 2
        half_y_size_pixels = y_size_pixels // 2
        
        planck_fk2 = {}
        planck_fk1 = {}
        rad = {}
        planck_bc1 = {}
        planck_bc2 = {}
        for band in self.bands:
            planck_fk2[band] = self.goes16_ds[band]["planck_fk2"].values
            planck_fk1[band] = self.goes16_ds[band]["planck_fk1"].values
            rad[band] = self.goes16_ds[band]["Rad"].values
            planck_bc1[band] = self.goes16_ds[band]["planck_bc1"].values
            planck_bc2[band] = self.goes16_ds[band]["planck_bc2"].values

        for i in range(center_lons.size):
            center_row = center_rows[i]
            center_col = center_cols[i]
            if center_row < half_y_size_pixels or center_col < half_x_size_pixels \
               or center_row+half_y_size_pixels >= self.y_g.shape[0] or center_col+half_x_size_pixels >= self.y_g.shape[1] :
                continue
            row_slice = slice(int(center_row - half_y_size_pixels), int(center_row + half_y_size_pixels))
            col_slice = slice(int(center_col - half_x_size_pixels), int(center_col + half_x_size_pixels))
            for b, band in enumerate(self.bands):
                if bt:
                    patches[i, b, :, :] = (planck_fk2[band] /
                                             np.log(planck_fk1[band] /
                                             rad[band][row_slice, col_slice] + 1) -
                                             planck_bc1[band]) / planck_bc2[band]
                else:
                    patches[i, b, :, :] = rad[band][row_slice, col_slice]
                lons[i,:,:] = self.lon[row_slice, col_slice]
                lats[i,:,:] = self.lat[row_slice, col_slice]
        
        del planck_fk2
        del planck_fk1
        del rad
        del planck_bc1
        del planck_bc2

        return patches,lons,lats


    def close(self):
        for band in self.bands:
            self.goes16_ds[band].close()
            del self.goes16_ds[band]


def extract_abi_patches(abi_path, patch_path, glm_grid_path, glm_file_date, bands,
                        lead_time, patch_x_length_pixels, patch_y_length_pixels, samples_per_time,
                        glm_file_freq="1D", max_pos_sample_ratio=0.5, glm_date_format="%Y%m%dT%H%M%S",
                        time_range_minutes=4, bt=False):
    """
    For a given set of gridded GLM counts, sample from the grids at each time step and extract ABI
    patches centered on the lightning grid cell.

    Args:
        abi_path (str): path to GOES-16 ABI data
        patch_path (str): Path to GOES-16 output patches
        glm_grid_path (str): Path to GLM grid files
        glm_file_date (:class:`pandas.Timestamp`): Day of GLM file being extracted
        bands (:class:`numpy.ndarray`, int): timeArray of band numbers
        lead_time (str): Lead time in pandas Timedelta units
        patch_x_length_pixels (int): Size of patch in x direction in pixels
        patch_y_length_pixels (int): Size of patch in y direction in pixels
        samples_per_time (int): Number of grid points to select without replacement at each timestep
        glm_file_freq (str): How ofter GLM files are use
        glm_date_format (str): How the GLM date is formatted
        time_range_minutes (int): Minutes before or after time in which GOES16 files are valid.
        bt (bool): Calculate brightness temperature instead of radiance

    Returns:

    """
    start_date_str = glm_file_date.strftime(glm_date_format)
    end_date_str = (glm_file_date + pd.Timedelta(glm_file_freq)).strftime(glm_date_format)
    glm_grid_file = join(glm_grid_path, "glm_grid_s{0}_e{1}.nc".format(start_date_str, end_date_str))
    if not exists(glm_grid_file):
        raise FileNotFoundError(glm_grid_file + " not found")
    glm_ds = xr.open_dataset(glm_grid_file)
    times = pd.DatetimeIndex(glm_ds["time"].values)
    lons = glm_ds["lon"]
    lats = glm_ds["lat"]
    counts = glm_ds["lightning_counts"]
    patches = np.zeros((times.size * samples_per_time, bands.size, patch_y_length_pixels, patch_x_length_pixels),
                       dtype=np.float32)
    patch_lons = np.zeros((times.size * samples_per_time, patch_y_length_pixels, patch_x_length_pixels),
                          dtype=np.float32)
    patch_lats = np.zeros((times.size * samples_per_time, patch_y_length_pixels, patch_x_length_pixels),
                          dtype=np.float32)
    flash_counts = np.zeros((times.size * samples_per_time), dtype=np.int32)
    grid_sample_indices = np.arange(lons.size, dtype=np.int32)
    max_pos_counts = int(samples_per_time * max_pos_sample_ratio)
    is_valid = np.ones((times.size * samples_per_time), dtype=bool)
    patch_times = []
    for t, time in enumerate(times):
        print(time, flush=True)
        patch_time = time - pd.Timedelta(lead_time)
        pos_count = np.count_nonzero(counts[t] > 0)
        pos_sample_size = np.minimum(pos_count, max_pos_counts)
        neg_sample_size = samples_per_time - pos_sample_size
        count_grid = counts[t].values
        if pos_sample_size > 0:
            pos_time_samples = np.random.choice(grid_sample_indices[count_grid.ravel() > 0], size=pos_sample_size,
                                                replace=False)
            neg_time_samples = np.random.choice(grid_sample_indices[count_grid.ravel() == 0], size=neg_sample_size,
                                                replace=False)
            time_samples = np.concatenate([pos_time_samples, neg_time_samples])
        else:
            time_samples = np.random.choice(grid_sample_indices, size=samples_per_time, replace=False)
        sample_rows, sample_cols = np.unravel_index(time_samples, lons.shape)
        patch_times.extend([time] * samples_per_time)
        try:
            goes16_abi_timestep = GOES16ABI(patch_time, bands, abi_path, time_range_minutes=time_range_minutes)
            for s in range(samples_per_time):
                flash_counts[t * samples_per_time + s] = count_grid[sample_rows[s], sample_cols[s]]
                patches[t * samples_per_time + s], \
                    patch_lons[t * samples_per_time + s], \
                    patch_lats[t * samples_per_time + s] = goes16_abi_timestep.extract_image_patch(lons[sample_rows[s], sample_cols[s]],
                                                                                lats[sample_rows[s], sample_cols[s]],
                                                                                patch_x_length_pixels,
                                                                                patch_y_length_pixels,
                                                                                bt=bt)
            goes16_abi_timestep.close()
            del goes16_abi_timestep
        except FileNotFoundError as fnfe:
            print(fnfe.args)
            is_valid[t*samples_per_time: t * samples_per_time + samples_per_time] = False 
    x_coords = np.arange(patch_x_length_pixels)
    y_coords = np.arange(patch_y_length_pixels)
    valid_patches = np.where(is_valid)[0]
    patch_num = np.arange(valid_patches.shape[0])
    glm_ds.close()
    del glm_ds
    patch_ds = xr.Dataset(data_vars={"abi": (("patch", "band", "y", "x"), patches[valid_patches]),
                                     "time": (("patch", ), pd.DatetimeIndex(patch_times)[valid_patches]),
                                     "lon": (("patch", "y", "x"), patch_lons[valid_patches]),
                                     "lat": (("patch", "y", "x"), patch_lats[valid_patches]),
                                     "flash_counts": (("patch", ), flash_counts[valid_patches])},
                          coords={"patch": patch_num,
                                  "y": y_coords, "x": x_coords, "band": bands})
    out_file = join(patch_path, "abi_patches_{0}.nc".format(glm_file_date.strftime(glm_date_format)))
    if not exists(patch_path):
        makedirs(patch_path)
    patch_ds.to_netcdf(out_file,
                       engine="netcdf4",
                       encoding={"abi": {"zlib": True}, "lon": {"zlib": True}, "lat": {"zlib": True},
                                 "flash_counts": {"zlib": True}})
    return 0


def extract_all_abi_patches(abi_path, patch_path, glm_grid_path, glm_file_date, bands,
                        lead_time, patch_x_length_pixels, patch_y_length_pixels, samples_per_time,
                        glm_file_freq="1D", glm_date_format="%Y%m%dT%H%M%S",
                        time_range_minutes=4, bt=False):
    """
    For a given set of gridded GLM counts, sample from the grids at each time step and extract ABI
    patches centered on the lightning grid cell.

    Args:
        abi_path (str): path to GOES-16 ABI data
        patch_path (str): Path to GOES-16 output patches
        glm_grid_path (str): Path to GLM grid files
        glm_file_date (:class:`pandas.Timestamp`): Day of GLM file being extracted
        bands (:class:`numpy.ndarray`, int): timeArray of band numbers
        lead_time (str): Lead time in pandas Timedelta units
        patch_x_length_pixels (int): Size of patch in x direction in pixels
        patch_y_length_pixels (int): Size of patch in y direction in pixels
        samples_per_time (int): Number of grid points to select without replacement at each timestep
        glm_file_freq (str): How ofter GLM files are use
        glm_date_format (str): How the GLM date is formatted
        time_range_minutes (int): Minutes before or after time in which GOES16 files are valid.
        bt (bool): Calculate brightness temperature instead of radiance

    Returns:

    """
    np.random.seed(100)
    start_date_str = glm_file_date.strftime(glm_date_format)
    end_date_str = (glm_file_date + pd.Timedelta(glm_file_freq)).strftime(glm_date_format)
    glm_grid_file = join(glm_grid_path, "glm_grid_s{0}_e{1}.nc".format(start_date_str, end_date_str))
    if not exists(glm_grid_file):
        raise FileNotFoundError(glm_grid_file + " not found")
    glm_ds = xr.open_dataset(glm_grid_file)
    times = pd.DatetimeIndex(glm_ds["time"].values)
    lons = glm_ds["lon"].values
    lats = glm_ds["lat"].values
    counts = glm_ds["lightning_counts"]
    patches = np.zeros((times.size*lats.size, bands.size, patch_y_length_pixels, patch_x_length_pixels),
                       dtype=np.float32)
    patch_lons = np.zeros((times.size*lats.size, patch_y_length_pixels, patch_x_length_pixels),
                          dtype=np.float32)
    patch_lats = np.zeros((times.size*lats.size, patch_y_length_pixels, patch_x_length_pixels),
                          dtype=np.float32)
    flash_counts = np.zeros((times.size*lats.size), dtype=np.int32)
    is_valid = np.ones((times.size * lats.size), dtype=bool)
    patch_times = []
    for t, time in enumerate(times):
        print(time, flush=True)
        patch_time = time - pd.Timedelta(lead_time)
        count_grid = counts[t].values
        patch_times.extend([time] * lats.size)
        try:
            goes16_abi_timestep = GOES16ABI(patch_time, bands, abi_path, time_range_minutes=time_range_minutes)
            flash_counts[t*lats.size:t*lats.size+lats.size] = count_grid.ravel()

            start_t = cpytime.time()
            patches[t*lats.size:t*lats.size+lats.size], \
                patch_lons[t*lats.size:t*lats.size+lats.size], \
                patch_lats[t*lats.size:t*lats.size+lats.size] = goes16_abi_timestep.extract_all_image_patchs(lons,
                                                                                lats,
                                                                                patch_x_length_pixels,
                                                                                patch_y_length_pixels,
                                                                                bt=bt)
            print("extract_all_image_patchs: %d" % (cpytime.time()-start_t))

            if np.all(np.isnan(patches[t*lats.size:t*lats.size+lats.size])):
                is_valid[t*lats.size: t * lats.size + lats.size] = False 

            goes16_abi_timestep.close()
            del goes16_abi_timestep
        except FileNotFoundError as fnfe:
            print(fnfe.args)
            is_valid[t*lats.size: t * lats.size + lats.size] = False 
    x_coords = np.arange(patch_x_length_pixels)
    y_coords = np.arange(patch_y_length_pixels)
    valid_patches = np.where(is_valid)[0]
    patch_num = np.arange(valid_patches.shape[0])
    glm_ds.close()
    del glm_ds
    patch_ds = xr.Dataset(data_vars={"abi": (("patch", "band", "y", "x"), patches[valid_patches]),
                                     "time": (("patch", ), pd.DatetimeIndex(patch_times)[valid_patches]),
                                     "lon": (("patch", "y", "x"), patch_lons[valid_patches]),
                                     "lat": (("patch", "y", "x"), patch_lats[valid_patches]),
                                     "flash_counts": (("patch", ), flash_counts[valid_patches])},
                          coords={"patch": patch_num,
                                  "y": y_coords, "x": x_coords, "band": bands})
    out_file = join(patch_path, "abi_patches_{0}.nc".format(glm_file_date.strftime(glm_date_format)))
    if not exists(patch_path):
        makedirs(patch_path)
    patch_ds.to_netcdf(out_file,
                       engine="netcdf4",
                       encoding={"abi": {"zlib": True}, "lon": {"zlib": True}, "lat": {"zlib": True},
                                 "flash_counts": {"zlib": True}})
    return 0


def regrid_imagery(image, x_image, y_image, x_regrid, y_regrid, image_proj, regrid_proj, spline_kws=None):
    """
    For a given image, regrid it to another projection using spline interpolation.

    Args:
        image:
        x_image:
        y_image:
        x_regrid:
        y_regrid:
        image_proj:
        regrid_proj:
        spline_kws:

    Returns:

    """
    if spline_kws is None:
        spline_kws = dict()
    x_regrid_image, y_regrid_image = transform(image_proj, regrid_proj, x_regrid.ravel(), y_regrid.ravel())
    rbs = RectBivariateSpline(x_image, y_image, image, **spline_kws)
    regridded_image = rbs.ev(x_regrid_image, y_regrid_image).reshape(x_regrid.shape)
    return regridded_image

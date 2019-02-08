import pandas as pd
import numpy as np
from glob import glob
from datetime import datetime
from pyproj import Proj, transform
from os.path import join
from scipy.interpolate import RectBivariateSpline
import xarray as xr


class GOES16ABI(object):
    """
    Handles data I/O and map projections for GOES-16 Advanced Baseline Imager data.

    Attributes:
        date (:class:`pandas.Timestamp`): Date of image slices
        channels (:class:`numpy.ndarray`): GOES-16 channels/bands to load
        path (str): Path to top level of GOES-16 ABI directory.
        time_range_minutes (int): interval in number of minutes to search for file that matches input time
        goes16_ds (`dict` of :class:`xarray.Dataset` objects): Datasets for each channel

    """
    def __init__(self, date, channels, path, time_range_minutes=2):
        self.date = pd.Timestamp(date)
        self.channels = np.array(channels, dtype=np.int32)
        self.path = path
        self.time_range_minutes = time_range_minutes
        self.goes16_ds = dict()
        self.channel_files = []
        for channel in channels:
            self.channel_files.append(self.goes16_abi_filename(channel))
            self.goes16_ds[channel] = xr.open_dataset(self.channel_files[-1])
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
    def abi_file_dates(files, file_date='s'):
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
        pd_date = self.date
        channel_files = sorted(glob(join(self.path, pd_date.strftime("%Y%m%d"),
                                         f"OR_ABI-L1b-RadC-M3C{channel:02d}_G16_*.nc")))
        channel_dates = self.abi_file_dates(channel_files)
        file_index = np.where(np.abs(pd_date - channel_dates) < pd.Timedelta(minutes=self.time_range_minutes))[0]
        if len(file_index) == 0:
            raise FileNotFoundError('No GOES-16 files within 2 minutes of ' + pd_date)
        else:
            filename = channel_files[file_index[0]]
        return filename

    def goes16_projection(self):
        """
        Create a Pyproj projection object with the projection information from a GOES-16 file.
        The geostationary map projection is described in the
        `PROJ <https://proj4.org/operations/projections/geos.html>`_ documentation.

        """
        goes16_ds = self.goes16_ds[self.channels.min()]
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
        goes16_ds = self.goes16_ds[self.channels.min()]
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

    def extract_image_patch(self, center_lon, center_lat, x_size_pixels, y_size_pixels):
        """


        Args:
            center_lon:
            center_lat:
            x_size_pixels:
            y_size_pixels:

        Returns:

        """
        center_x, center_y = self.proj(center_lon, center_lat)
        center_row = np.argmin(np.abs(self.y - center_y))
        center_col = np.argmin(np.abs(self.x - center_x))
        row_slice = slice(int(center_row - y_size_pixels // 2), int(center_row + y_size_pixels // 2))
        col_slice = slice(int(center_col - x_size_pixels // 2), int(center_col + x_size_pixels // 2))
        patch = np.zeros((1, y_size_pixels, x_size_pixels, self.channels.size), dtype=np.float32)
        for c, channel in enumerate(self.channels):
            patch[0, :, :, c] = self.goes16_ds[channel]["Rad"][row_slice, col_slice].values
        return patch

    def close(self):
        for channel in self.channels:
            self.goes16_ds[channel].close()
            del self.goes16_ds[channel]


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

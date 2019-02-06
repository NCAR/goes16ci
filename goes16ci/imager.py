import pandas as pd
import numpy as np
import xarray as xr
from glob import glob
from datetime import datetime
from pyproj import Proj
from os.path import join, exists


def abi_file_dates(files, file_date='c'):
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


def goes16_abi_filename(date, channel, path, time_range_minutes=2):
    """
    Given a path to a dataset of GOES-16 files, find the netCDF file that matches the expected
    date and channel, or band number.

    The GOES-16 path should point to a directory containing a series of directories named by
    valid date in %Y%m%d format. Each directory should contain Level 1 CONUS sector files.


    Args:
        date (:class:`str`, :class:`datetime.datetime`, or :class:`pandas.Timestamp`): contains the date of the image
        channel (int): GOES-16 ABI `channel <https://www.goes-r.gov/mission/ABI-bands-quick-info.html>`_.
        path (str): Path to top-level directory containing GOES-16 netCDF files.

    Returns:
        str: full path to requested GOES-16 file
    """
    pd_date = pd.Timestamp(date)
    channel_files = sorted(glob(join(path, pd_date.strftime("%Y%m%d"), f"OR_ABI-L1b-RadC-M3C{channel:02d}_G16_*.nc")))

    channel_dates = abi_file_dates(channel_files)
    file_index = np.where(np.abs(pd_date - channel_dates) < pd.Timedelta(minutes=time_range_minutes))[0]
    if len(file_index) == 0:
        raise FileNotFoundError('No GOES-16 files within 2 minutes of ' + pd_date)
    else:
        filename = channel_files[file_index[0]]
    return filename


def goes16_projection(goes16_ds):
    """
    Create a Pyproj projection object with the projection information from a GOES-16 file.
    The geostationary map projection is described in the
    `PROJ <https://proj4.org/operations/projections/geos.html>`_ documentation.

    Args:
        goes16_ds (:class:`xarray.Dataset`): GOES-16 dataset object containing projection information

    Returns:
        pyproj.Proj: Proj object with geostationary projection.
    """
    proj_dict = dict(proj="geos",
                     h=goes16_ds["goes_imager_projection"].attrs["perspective_point_height"],
                     lon_0=goes16_ds["goes_imager_projection"].attrs["longitude_of_projection_origin"],
                     sweep=goes16_ds["goes_imager_projection"].attrs["sweep_angle_axis"])
    return Proj(projparams=proj_dict)


def sat_coordinates(goes16_ds):
    """
    Calculate the geostationary projection x and y coordinates in m for each
    pixel in the image.

    Args:
        goes16_ds (:class:`xarray.Dataset`): GOES-16 image dataset

    Returns:
        :class:`numpy.ndarray`, :class:`numpy.ndarray` : x, y

    """
    sat_height = goes16_ds["goes_imager_projection"].attrs["perspective_point_height"]
    x_h = goes16_ds["x"] * sat_height
    y_h = goes16_ds["y"] * sat_height
    x_g, y_g = np.meshgrid(x_h, y_h)
    return x_g, y_g


def lon_lat_coords(goes16_ds, projection):
    """
    Calculate longitude and latitude coordinates for each point in the GOES-16
    image.

    Args:
        goes16_ds (:class:`xarray.Dataset`): GOES-16 image dataset
        projection (pyproj.Proj): GOES-16 map projection

    Returns:
        :class:`numpy.ndarray`, :class:`numpy.ndarray` : longitudes, latitudes
    """
    x_g, y_g = sat_coordinates(goes16_ds)
    lons, lats = projection(x_g, y_g, inverse=True)
    lons[lons > 1e10] = np.nan
    lats[lats > 1e10] = np.nan
    return lons, lats

import numpy as np
import pandas as pd
from goes16ci.lightning import create_glm_grids
from goes16ci.imager import extract_abi_patches
from dask.distributed import LocalCluster, Client, wait
import argparse
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Config yaml file")
    parser.add_argument("-n", "--nprocs", type=int, default=1, help="Number of processes")
    parser.add_argument("-l", "--glm", action="store_true", help="Create GLM grids")
    parser.add_argument("-a", "--abi", action="store_true", help="Sample ABI Patches")
    args = parser.parse_args()
    with open(args.config, "r") as config_file:
        config = yaml.load(config_file)
    cluster = LocalCluster(n_workers=args.nprocs, processes=True, threads_per_worker=1)
    client = Client(cluster, asynchronous=True)
    print(cluster, flush=True)
    print(client, flush=True)
    if args.glm:
        glm_config = config["glm"]
        glm_path = glm_config["glm_path"]
        grid_path = glm_config["grid_path"]
        start_date = pd.Timestamp(glm_config["start_date"])
        end_date = pd.Timestamp(glm_config["end_date"])
        file_freq = glm_config["file_freq"]
        glm_file_dates = pd.DatetimeIndex(pd.date_range(start=start_date, end=end_date, freq=file_freq))
        grid_freq = glm_config["grid_freq"]
        grid_proj_params = glm_config["grid_proj_params"]
        dx_km = glm_config["dx_km"]
        x_extent_km = glm_config["x_extent_km"]
        y_extent_km = glm_config["y_extent_km"]
        glm_jobs = []
        for date in glm_file_dates:
            glm_jobs.append(client.submit(create_glm_grids, glm_path, grid_path, date, date + pd.Timedelta(file_freq),
                                          grid_freq, grid_proj_params, dx_km, x_extent_km, y_extent_km))
        wait(glm_jobs)
        glm_results = client.gather(glm_jobs)
        del glm_jobs[:]
    if args.abi:
        abi_config = config["abi"]
        abi_path = abi_config["abi_path"]
        patch_path = abi_config["patch_path"]
        glm_grid_path = abi_config["glm_grid_path"]
        start_date = pd.Timestamp(abi_config["start_date"])
        end_date = pd.Timestamp(abi_config["end_date"])
        bands = np.array(abi_config["bands"])
        file_freq = abi_config["file_freq"]
        lead_time = abi_config["lead_time"]
        patch_x_length_pixels = abi_config["patch_x_length_pixels"]
        patch_y_length_pixels = abi_config["patch_y_length_pixels"]
        samples_per_time = abi_config["samples_per_time"]
        abi_file_dates = pd.TimedeltaIndex(start=start_date, end=end_date, freq=file_freq)
        abi_jobs = []
        for date in abi_file_dates:
            abi_jobs.append(client.submit(extract_abi_patches, abi_path, patch_path, glm_grid_path, ))
        wait(abi_jobs)
        abi_results = client.gather(abi_jobs)
        del abi_jobs[:]
    client.close()
    return


def glm_single_date():
    glm_path = "/Users/dgagne/data/goes16_nc/GLM-L2_LCFA"
    out_path = '/Users/dgagne/data/goes16_glm_grids'
    start_date = pd.Timestamp("2018-05-01 03:00:00")
    end_date = pd.Timestamp("2018-05-02")
    out_freq = "60min"
    grid_proj_params = {'proj': 'lcc', 'lon_0': -90.0, 'lat_0': 35.0, 'lat_1': 25.0, 'lat_2': 50.0}
    dx_km = 10.0
    x_extent_km = 1000.0
    y_extent_km = 2000.0
    flash_grid = create_glm_grids(glm_path, out_path, start_date, end_date, out_freq, grid_proj_params,
                                  dx_km, x_extent_km, y_extent_km)


if __name__ == "__main__":
    main()

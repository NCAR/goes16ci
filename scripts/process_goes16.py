import numpy as np
import pandas as pd
from goes16ci.lightning import create_glm_grids
from dask.distributed import LocalCluster, Client
import argparse
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", required=True, help="Config yaml file")
    parser.add_argument("-p", "--procs", type=int, default=1, help="Number of processes")
    parser.add_argument("-l", "--glm", action="store_true", help="Create GLM grids")
    parser.add_argument("-p", "--patch", action="store_true", help="Sample GOES-16 patches")
    args = parser.parse_args()
    with open(args.config, "r") as config_file:
        config = yaml.load(config_file)
    cluster = LocalCluster(n_workers=args.procs)
    client = Client(cluster)

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
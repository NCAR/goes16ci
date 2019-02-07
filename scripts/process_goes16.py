import numpy as np
import pandas as pd
from goes16ci.lightning import create_glm_grids


def main():
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
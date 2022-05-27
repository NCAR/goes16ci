import pandas as pd
import numpy as np
#from goes16ci.lightning import create_glm_grids
from nldn_lightning import create_nldn_grids
from goes16_abi_unet import extract_abi_patches
from dask.distributed import LocalCluster, Client, as_completed, wait
import argparse
import yaml
import traceback
from os.path import exists
from os import makedirs
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Config yaml file")
    parser.add_argument("-n", "--nprocs", type=int, default=1, help="Number of processes")
    parser.add_argument("-l", "--nldn", action="store_true", help="Create NLDN grids")
    parser.add_argument("-a", "--abi", action="store_true", help="Sample ABI Patches")
    args = parser.parse_args()
    with open("goes16_process_config.yml", "r") as config_file:
        config = yaml.safe_load(config_file) 
    cluster = LocalCluster(n_workers=1, processes=True, threads_per_worker=1)
    client = Client(cluster)
    print(cluster, flush=True)
    print(client, flush=True)
    glm_config = config["glm"]
    glm_path = glm_config["glm_path"]
    if args.nldn:
        print("Starting lightning", flush=True)
        nldn_config = config["nldn"]
        nldn_path = nldn_config["nldn_path"]
        grid_path = nldn_config["grid_path"]
        start_date = pd.Timestamp(nldn_config["start_date"])
        end_date = pd.Timestamp(nldn_config["end_date"])
        file_freq = nldn_config["file_freq"]
        nldn_file_dates = pd.DatetimeIndex(pd.date_range(start=start_date, end=end_date, freq=file_freq))
        grid_freq = nldn_config["grid_freq"]
        grid_proj_params = nldn_config["grid_proj_params"]
        dx_km = glm_config["dx_km"]
        x_extent_km = nldn_config["x_extent_km"]
        y_extent_km = nldn_config["y_extent_km"]
        if not exists(grid_path):
            makedirs(grid_path)
        nldn_jobs = []
        for date in nldn_file_dates:
            print(date, flush=True)
            glm_jobs.append(client.submit(create_nldn_grids, nldn_path, grid_path, date, date + pd.Timedelta(file_freq),
                                          grid_freq, grid_proj_params, dx_km, x_extent_km, y_extent_km))
        for nldn_job in as_completed(nldn_jobs):
            res = nldn_job.result()
            if nldn_job.status == "error":
                traceback.format_tb(res[-1])
        #wait(glm_jobs)
        #glm_results = client.gather(glm_jobs)
        del nldn_jobs[:]
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
        time_range_minutes = abi_config["time_range_minutes"]
        bt = bool(abi_config["bt"])
        if not exists(patch_path):
            makedirs(patch_path)
        abi_file_dates = pd.DatetimeIndex(pd.date_range(start=start_date, end=end_date, freq=file_freq))
        abi_jobs = []
        if args.nprocs :
            for date in abi_file_dates:
                abi_jobs.append(client.submit(extract_abi_patches, abi_path, patch_path, glm_grid_path, glm_path, date, bands,
                                    lead_time, patch_x_length_pixels, patch_y_length_pixels, samples_per_time,
                                    time_range_minutes=time_range_minutes, glm_file_freq=file_freq, bt=bt))
        # for abi_job in as_completed(abi_jobs):
        #     res = abi_job.result()
        #     if abi_job.status == "error":
        #         print(traceback.format_tb(res[-1]),flush=True)
            wait(abi_jobs)
            abi_results = client.gather(abi_jobs)
            del abi_jobs[:]
        else:
            for date in abi_file_dates:
                extract_abi_patches(abi_path, patch_path, glm_grid_path, glm_path, date, bands,
                                    lead_time, patch_x_length_pixels, patch_y_length_pixels, samples_per_time,
                                    time_range_minutes=time_range_minutes, glm_file_freq=file_freq, bt=bt)
            
    
    client.close()
    return

if __name__ == "__main__":
    main()
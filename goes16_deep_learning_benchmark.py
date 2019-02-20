import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.extend([dir_path])
import yaml
import pandas as pd
import numpy as np
from goes16ci.data import load_data_parallel, load_data_serial
from goes16ci.models import train_conv_net_cpu, train_conv_net_gpu, MinMaxScaler2D
from timeit import default_timer as timer
import argparse
import logging
from datetime import datetime


def main():
    # read config file
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="benchmark_config.yml", help="Config yaml file")
    args = parser.parse_args()
    with open(args.config, "r") as config_file:
        config = yaml.load(config_file)
    logging.basicConfig(stream=sys.stdout, level=config["log_level"])
    benchmark_data = dict()
    # load data serial
    logging.info("Begin serial load data")
    benchmark_data["load_data_serial"] = {}
    benchmark_data["load_data_serial"]["start"] = timer()
    all_data, all_counts, all_time = load_data_serial(config["data_path"])
    benchmark_data["load_data_serial"]["end"] = timer()
    benchmark_data["load_data_serial"]["duration"] = benchmark_data["load_data_serial"]["end"] - benchmark_data["load_data_serial"]["start"]
    for v in range(all_data.shape[-1]):
        all_data[:, :, :, v][np.isnan(all_data[:, :, :, v])] = np.nanmin(all_data[:, :, :, v])
    #del all_data, all_counts, all_time
    # load data parallel
    logging.info("Begin parallel load data")
    benchmark_data["load_data_parallel"] = {}
    benchmark_data["load_data_parallel"]["processes"] = config["parallel_processes"]
    benchmark_data["load_data_parallel"]["start"] = timer()
   # all_data, all_counts, all_time = load_data_parallel(config["data_path"], config["parallel_processes"])
    benchmark_data["load_data_parallel"]["end"] = timer()
    benchmark_data["load_data_parallel"]["duration"] = benchmark_data["load_data_parallel"]["end"] - \
        benchmark_data["load_data_parallel"]["start"]
    # Split training and validation data
    logging.info("Split training and testing data")
    train_indices = np.where(all_time < pd.Timestamp(config["split_date"]))[0]
    val_indices = np.where(all_time >= pd.Timestamp(config["split_date"]))[0]
    train_data = all_data[train_indices].astype(config["dtype"])
    val_data = all_data[val_indices].astype(config["dtype"])
    train_counts = np.where(all_counts[train_indices] > 0, 1, 0).astype(config["dtype"])
    val_counts = np.where(all_counts[val_indices] > 0, 1, 0).astype(config["dtype"])
    # Rescale training and validation data
    scaler = MinMaxScaler2D()
    train_data_scaled = scaler.fit_transform(train_data)
    val_data_scaled = scaler.transform(val_data)
    # CPU training
    if config["cpu"]:
        logging.info("CPU Training")
        benchmark_data["cpu_training"] = {}
        benchmark_data["cpu_training"]["start"] = timer()
        train_conv_net_cpu(train_data_scaled, train_counts, val_data_scaled, val_counts, config["conv_net_parameters"],
                           config["num_cpus"], config["random_seed"])
        benchmark_data["cpu_training"]["end"] = timer()
        benchmark_data["cpu_training"]["duration"] = benchmark_data["cpu_training"]["end"] - \
                                                           benchmark_data["cpu_training"]["start"]

    # CPU inference

    # Multi GPU Training
    if config["multi_gpu"]:
        logging.info("Multi GPU Training")
        benchmark_data["gpu_m_training"] = {}
        benchmark_data["gpu_m_training"]["start"] = timer()
        train_conv_net_gpu(train_data_scaled, train_counts, 
                           val_data_scaled, val_counts, config["conv_net_parameters"],
                           config["num_gpus"], config["random_seed"], dtype=config["dtype"])
        benchmark_data["gpu_m_training"]["end"] = timer()
        benchmark_data["gpu_m_training"]["duration"] = benchmark_data["gpu_m_training"]["end"] - \
                                                           benchmark_data["gpu_m_training"]["start"]


    # Single GPU Training
    if config["single_gpu"]:
        logging.info("Single GPU Training")
        benchmark_data["gpu_1_training"] = {}
        benchmark_data["gpu_1_training"]["start"] = timer()
        train_conv_net_gpu(train_data_scaled, train_counts, 
                           val_data_scaled, val_counts, config["conv_net_parameters"],
                           1, config["random_seed"], dtype=config["dtype"])
        benchmark_data["gpu_1_training"]["end"] = timer()
        benchmark_data["gpu_1_training"]["duration"] = benchmark_data["gpu_1_training"]["end"] - \
                                                           benchmark_data["gpu_1_training"]["start"]
    # Single GPU Inference

        # Multi GPU Inference

    # Save benchmark data
    output_filename = "./goes_benchmark_data_{0}.yml".format(datetime.utcnow().strftime("%Y%m%d_%H%M%S"))
    logging.info("Saving benchmark data to {output_filename}".format(output_filename=output_filename))
    with open(output_filename, "w") as output_file:
        yaml.dump(benchmark_data, output_file)
    return


if __name__ == "__main__":
    main()

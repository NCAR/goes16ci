import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.extend([dir_path])
import yaml
import tensorflow as tf
import pandas as pd
import numpy as np
from os.path import exists, join
from goes16ci.data import load_data_serial
from goes16ci.models import train_conv_net_cpu, train_conv_net_gpu, MinMaxScaler2D
from goes16ci.monitor import Monitor, start_timing, end_timing, calc_summary_stats, get_gpu_names, get_gpu_topo, get_cuda_version
import argparse
import logging
from datetime import datetime
import platform
from multiprocessing import Pipe, Process
import traceback


def main():
    # read config file
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="benchmark_config.yml", help="Config yaml file")
    args = parser.parse_args()
    with open(args.config, "r") as config_file:
        config = yaml.load(config_file, Loader=yaml.Loader) 
    out_path = config["out_path"]
    logging.basicConfig(stream=sys.stdout, level=config["log_level"])
    print(repr(config["out_path"]))
    benchmark_data = dict()
    print(config["out_path"], type(config["out_path"]))
    # load data serial
    benchmark_data["config"] = config
    benchmark_data["system"] = dict()
    benchmark_data["system"]["hostname"] = platform.node()
    benchmark_data["system"]["platform"] = platform.platform()
    benchmark_data["system"]["python_version"] = platform.python_version()
    benchmark_data["system"]["python_compiler"] = platform.python_compiler()
    benchmark_data["system"]["tensorflow_version"] = tf.__version__
    benchmark_data["system"]["gpus"] = get_gpu_names()
    benchmark_data["system"].update(**get_cuda_version())
    benchmark_data["system"]["gpu_topology"] = get_gpu_topo()
    logging.info("Begin serial load data")
    all_data, all_counts, all_time = load_data_serial(config["data_path"])
    if not exists(config["out_path"]):
        os.makedirs(config["out_path"])
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
    train_data_scaled = 1.0 - scaler.fit_transform(train_data)
    val_data_scaled = 1.0 - scaler.transform(val_data)
    print(scaler.scale_values)
    # Start monitor process
    parent_p, child_p = Pipe()
    dl_monitor = Monitor(child_p)
    monitor_proc = Process(target=dl_monitor.run)
    monitor_proc.start()
    try:
        # CPU training
        if config["cpu"]:
            logging.info("CPU Training")
            block_name = "cpu_training"
            start_timing(benchmark_data, block_name, parent_p, out_path)
            train_conv_net_cpu(train_data_scaled, train_counts, val_data_scaled, val_counts, config["conv_net_parameters"],
                            config["num_cpus"], config["random_seed"])
            end_timing(benchmark_data, block_name, parent_p, out_path)
        # CPU inference

        # Multi GPU Training
        if config["multi_gpu"]:
            block_name = "gpu_{0:02d}_training".format(config["num_gpus"])
            logging.info("Multi GPU Training")
            start_timing(benchmark_data, block_name, parent_p, out_path)
            train_conv_net_gpu(train_data_scaled, train_counts, 
                            val_data_scaled, val_counts, config["conv_net_parameters"],
                            config["num_gpus"], config["random_seed"], dtype=config["dtype"])
            end_timing(benchmark_data, block_name, parent_p, out_path)
        # Single GPU Training
        if config["single_gpu"]:
            logging.info("Single GPU Training")
            block_name = "gpu_{0:02d}_training".format(1)
            start_timing(benchmark_data, block_name, parent_p, out_path)
            train_conv_net_gpu(train_data_scaled, train_counts, 
                            val_data_scaled, val_counts, config["conv_net_parameters"],
                            1, config["random_seed"], dtype=config["dtype"])
            end_timing(benchmark_data, block_name, parent_p, out_path)

        # Single GPU Inference

            # Multi GPU Inference

        # Save benchmark data
        parent_p.send("exit")
        monitor_proc.join()

        output_filename = str(join(config["out_path"], "goes_benchmark_data_{0}.yml".format(datetime.utcnow().strftime("%Y%m%d_%H%M%S"))))
        logging.info("Saving benchmark data to {output_filename}".format(output_filename=output_filename))
        with open(output_filename, "w") as output_file:
            yaml.dump(benchmark_data, output_file, Dumper=yaml.Dumper)
    except Exception as e:
        logging.error(traceback.format_exc())
        sys.exit()
    return


if __name__ == "__main__":
    main()

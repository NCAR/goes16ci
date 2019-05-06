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
from goes16ci.monitor import Monitor, start_timing, end_timing, calc_summary_stats
from time import perf_counter, process_time
import argparse
import logging
from datetime import datetime
import platform
from multiprocessing import Pipe, Process


def main():
    # read config file
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="benchmark_config.yml", help="Config yaml file")
    args = parser.parse_args()
    with open(args.config, "r") as config_file:
        config = yaml.load(config_file, Loader=yaml.SafeLoader)
    logging.basicConfig(stream=sys.stdout, level=config["log_level"])
    benchmark_data = dict()
    # load data serial
    benchmark_data["system"] = dict()
    benchmark_data["system"]["platform"] = platform.platform()
    benchmark_data["system"]["python_version"] = platform.python_version()
    benchmark_data["system"]["python_compiler"] = platform.python_compiler()
    benchmark_data["system"]["tensorflow_version"] = tf.__version__
    parent_p, child_p = Pipe()
    dl_monitor = Monitor(child_p)
    monitor_proc = Process(target=dl_monitor.run)
    monitor_proc.start()
    logging.info("Begin serial load data")
    benchmark_data["load_data_serial"] = {}
    benchmark_data["load_data_serial"]["elapsed_start"] = perf_counter()
    benchmark_data["load_data_serial"]["process_start"] = process_time()
    all_data, all_counts, all_time = load_data_serial(config["data_path"])
    benchmark_data["load_data_serial"]["elapsed_end"] = perf_counter()
    benchmark_data["load_data_serial"]["process_end"] = process_time()
    benchmark_data["load_data_serial"]["elapsed_duration"] = benchmark_data["load_data_serial"]["elapsed_end"] - benchmark_data["load_data_serial"]["elapsed_start"]
    benchmark_data["load_data_serial"]["process_duration"] = benchmark_data["load_data_serial"]["process_end"] - benchmark_data["load_data_serial"]["process_start"]
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
    # CPU training
    if config["cpu"]:
        logging.info("CPU Training")
        start_timing(benchmark_data, "cpu_training")
        train_conv_net_cpu(train_data_scaled, train_counts, val_data_scaled, val_counts, config["conv_net_parameters"],
                           config["num_cpus"], config["random_seed"])
        end_timing(benchmark_data, "cpu_training")


    # CPU inference

    # Multi GPU Training
    if config["multi_gpu"]:
        logging.info("Multi GPU Training")
        parent_p.send("start " + join(config["out_path"], "gpu_m_training_stats.csv"))
        start_timing(benchmark_data, "gpu_m_training")
        train_conv_net_gpu(train_data_scaled, train_counts, 
                           val_data_scaled, val_counts, config["conv_net_parameters"],
                           config["num_gpus"], config["random_seed"], dtype=config["dtype"])
        end_timing(benchmark_data, "gpu_m_training")
        parent_p.send("stop")
        calc_summary_stats(benchmark_data, "gpu_m_training", join(config["out_path"], "gpu_m_training_stats.csv"))
    # Single GPU Training
    if config["single_gpu"]:
        logging.info("Single GPU Training")
        parent_p.send("start " + join(config["out_path"], "gpu_1_training_stats.csv"))
        start_timing(benchmark_data, "gpu_1_training")
        train_conv_net_gpu(train_data_scaled, train_counts, 
                           val_data_scaled, val_counts, config["conv_net_parameters"],
                           1, config["random_seed"], dtype=config["dtype"])
        end_timing(benchmark_data, "gpu_1_training")
        parent_p.send("stop")
        calc_summary_stats(benchmark_data, "gpu_1_training", join(config["out_path"], "gpu_1_training_stats.csv"))

    # Single GPU Inference

        # Multi GPU Inference

    # Save benchmark data
    parent_p.send("exit")
    monitor_proc.join()

    output_filename = join(config["out_path"], "goes_benchmark_data_{0}.yml".format(datetime.utcnow().strftime("%Y%m%d_%H%M%S")))
    logging.info("Saving benchmark data to {output_filename}".format(output_filename=output_filename))
    with open(output_filename, "w") as output_file:
        yaml.dump(benchmark_data, output_file, Dumper=yaml.SafeDumper)
    return




if __name__ == "__main__":
    main()

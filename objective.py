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
from goes16ci.monitor import Monitor, start_timing, end_timing, get_gpu_names, get_gpu_topo, get_cuda_version, get_cudnn_version, get_nccl_version
import argparse
import logging
from datetime import datetime
import platform
from multiprocessing import Pipe, Process
import traceback
from aimlutils.hyper_opt.base_objective import *
import optuna


def main(config,trial):
    # read config file 
    #with open(config, "r") as config_file:
     #   config = yaml.load(config_file, Loader=yaml.Loader) 
    out_path = config["out_path"]
    logging.basicConfig(stream=sys.stdout, level=config["log_level"])
    benchmark_data = dict()
    # load data serial
    benchmark_data["config"] = config
    benchmark_data["system"] = dict()
    benchmark_data["system"]["hostname"] = platform.node()
    benchmark_data["system"]["platform"] = platform.platform()
    benchmark_data["system"]["python_version"] = platform.python_version()
    benchmark_data["system"]["python_compiler"] = platform.python_compiler()
    benchmark_data["system"]["tensorflow_version"] = tf.__version__
    benchmark_data["system"]["gpus"] = get_gpu_names()
    has_gpus = True
    if len(benchmark_data["system"]["gpus"]) == 0:
        has_gpus = False
    #benchmark_data["system"].update(**get_cuda_version())
    benchmark_data["system"]["cudnn_version"] = get_cudnn_version()
    benchmark_data["system"]["nccl_version"] = get_nccl_version()
    benchmark_data["system"]["gpu_topology"] = get_gpu_topo()
    for k, v in benchmark_data["system"].items():
        print(k)
        print(v)
    logging.info("Begin serial load data")
    if "start_date" in config.keys():
        all_data, all_counts, all_time = load_data_serial(config["data_path"], start_date=config["start_date"],
                                                          end_date=config["end_date"])
    else:
        all_data, all_counts, all_time = load_data_serial(config["data_path"])
    if not exists(config["out_path"]):
        os.makedirs(config["out_path"])
    if "scale_batch_size" in config.keys():
        scale_batch_size = config["scale_batch_size"]
    else:
        scale_batch_size = 1
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
    #print out scaler values to csv for possible patch inference
    np.savetxt("scaler_values.csv", scaler.scale_values)
    # Start monitor process
    parent_p, child_p = Pipe()
    dl_monitor = Monitor(child_p)
    monitor_proc = Process(target=dl_monitor.run)
    monitor_proc.start()
    batch_loss = None
    epoch_loss = None
    try:
        # CPU training
        if config["cpu"]:
            logging.info("CPU Training")
            block_name = "cpu_training"
            start_timing(benchmark_data, block_name, parent_p, out_path)
            epoch_times, batch_loss, epoch_loss = train_conv_net_cpu(train_data_scaled, train_counts,
                                             val_data_scaled, val_counts, config["conv_net_parameters"],
                                             config["num_cpus"], config["random_seed"],trial=trial)
            end_timing(benchmark_data, epoch_times, block_name, parent_p, out_path)
            benchmark_data[block_name]["batch_loss"] = batch_loss
            benchmark_data[block_name]["epoch_loss"] = epoch_loss

        # CPU inference

        # Multi GPU Training
        if config["multi_gpu"] and has_gpus:
            gpu_nums = np.array([2, 4, 8])
            gpu_nums = gpu_nums[gpu_nums <= np.minimum(config["num_gpus"], len(benchmark_data["system"]["gpus"]))]
            for gpu_num in gpu_nums:
                block_name = "gpu_{0:02d}_training".format(gpu_num)
                logging.info("Multi GPU Training {0:02d}".format(gpu_num))
                start_timing(benchmark_data, block_name, parent_p, out_path)
                epoch_times, batch_loss, epoch_loss = train_conv_net_gpu(train_data_scaled, train_counts,
                                                 val_data_scaled, val_counts, config["conv_net_parameters"],
                                                 gpu_num, config["random_seed"], dtype=config["dtype"], scale_batch_size=scale_batch_size,trial=trial)
                end_timing(benchmark_data, epoch_times, block_name, parent_p, out_path)
                benchmark_data[block_name]["batch_loss"] = batch_loss
                benchmark_data[block_name]["epoch_loss"] = epoch_loss
                
        # Single GPU Training
        if config["single_gpu"] and has_gpus:
            logging.info("Single GPU Training")
            block_name = "gpu_{0:02d}_training".format(1)
            start_timing(benchmark_data, block_name, parent_p, out_path)
            try:
                epoch_times, batch_loss, epoch_loss = train_conv_net_gpu(train_data_scaled, train_counts,
                                val_data_scaled, val_counts, config["conv_net_parameters"],
                                1, config["random_seed"], dtype=config["dtype"],trial=trial)
            except Exception as E:
                logging.warning("{}".format(E))
                if "val_loss" in str(E):
                    raise OSError("prune")
                else:
                    raise E
            end_timing(benchmark_data, epoch_times, block_name, parent_p, out_path)
            benchmark_data[block_name]["batch_loss"] = batch_loss
            benchmark_data[block_name]["epoch_loss"] = epoch_loss
    
        # Save benchmark data
        parent_p.send("exit")
        monitor_proc.join()
        output_filename = str(join(config["out_path"], "goes_benchmark_data_{0}.yml".format(datetime.utcnow().strftime("%Y%m%d_%H%M%S"))))
        logging.info("Saving benchmark data to {output_filename}".format(output_filename=output_filename))
        with open(output_filename, "w") as output_file:
            yaml.dump(benchmark_data, output_file,Dumper=yaml.Dumper)
        print_summary(benchmark_data)
    except Exception as e:
        logging.error(traceback.format_exc())
        if "prune" in str(e):
            raise optuna.TrialPruned()
        else:
            parent_p.send("exit")
            monitor_proc.join()
            sys.exit()
    return epoch_loss


def print_summary(benchmark_data):
    logging.info("*** GOES Summary ***")
    if "cpu_training" in benchmark_data.keys():
        logging.info("CPU Training")
        logging.info("Elapsed: {0:0.2f}".format(benchmark_data["cpu_training"]["elapsed_duration"]))
        logging.info("Epoch: {0:0.2f}".format(benchmark_data["cpu_training"]["epoch_duration"]))
        logging.info("Epoch/Elapsed: {0:0.3f}".format(benchmark_data["cpu_training"]["epoch_duration"]
                                                      / benchmark_data["cpu_training"]["elapsed_duration"]))
        logging.info("\n")
    for gpu in [1, 2, 4, 8]:
        block_name = "gpu_{0:02d}_training".format(gpu)
        if block_name in benchmark_data.keys():
            logging.info("{0:02d} GPU Training".format(gpu))
            logging.info("Elapsed: {0:0.2f}".format(benchmark_data[block_name]["elapsed_duration"]))
            logging.info("Epoch: {0:0.2f}".format(benchmark_data[block_name]["epoch_duration"]))
            logging.info("Epoch/Elapsed: {0:0.3f}".format(benchmark_data[block_name]["epoch_duration"]
                                                          / benchmark_data[block_name]["elapsed_duration"]))
            logging.info("\n")

    return

class Objective(BaseObjective):

    def __init__(self, study, config, metric = "val_loss", device = "cpu"):

        # Initialize the base class
        BaseObjective.__init__(self, study, config, metric, device)

    def train(self, trial, conf):

        # Make any custom edits to the model conf before using it to train a model.
        #conf = custom_updates(trial, conf)

        result = main(conf,trial)
        results_dictionary = {
            "val_loss": result[-1]
        }
        return results_dictionary

if __name__ == "__main__":
    main()
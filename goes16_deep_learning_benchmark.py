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
import pickle
from sklearn.utils import class_weight


def main():
    # read config file 
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="benchmark_config_default.yml", help="Config yaml file")
    args = parser.parse_args()
    with open(args.config, "r") as config_file:
        config = yaml.load(config_file, Loader=yaml.Loader) 
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
    print("ALL COUNTS=",all_counts)
    print("all counts shape=", all_counts.shape) 
    train_indices = np.where(all_time < pd.Timestamp(config["split_date"]))[0]
    val_indices = np.where(all_time >= pd.Timestamp(config["split_date"]))[0]
    train_data = all_data[train_indices].astype(config["dtype"])
    val_data = all_data[val_indices].astype(config["dtype"])
    train_counts = all_counts[train_indices].astype(config["dtype"])
    #train_counts = np.where(all_counts[train_indices] > 0, 1, 0).astype(config["dtype"])
    print("Train counts =",train_counts)
    print("Train counts shape=", train_counts.shape)
    print("Train counts 1's=",len([x for x in train_counts if x == 1.0]))
    print("Train counts 2's=",len([y for y in train_counts if y == 0.0]))
    val_counts = np.where(all_counts[val_indices] > 0, 1, 0).astype(config["dtype"])
    # Rescale training and validation data
    #print("ALL DATA=", all_data)
    scaler = MinMaxScaler2D()
    train_data_scaled = 1.0 - scaler.fit_transform(train_data)
    val_data_scaled = 1.0 - scaler.transform(val_data)
    #save out to parquet files
    with open('train_data_scaled.pkl','wb') as f:
        pickle.dump(train_data_scaled, f)
    with open('train_counts.pkl','wb') as f:
        pickle.dump(train_counts, f)
    print(scaler.scale_values)
    scaler.scale_values.to_csv("scale_values.csv")
    # Start monitor process
    #train_ints = [t.argmax() for t in train_data_scaled]
    #print("Training Data==",train_data)
    class_weights = class_weight.compute_class_weight('balanced',np.unique(train_counts),train_counts)
    np.savetxt("class_weights.csv",class_weights, delimiter=",")
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
                                             config["num_cpus"], config["random_seed"],class_weights)
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
                                                 gpu_num, config["random_seed"], dtype=config["dtype"], scale_batch_size=scale_batch_size)
                end_timing(benchmark_data, epoch_times, block_name, parent_p, out_path)
                benchmark_data[block_name]["batch_loss"] = batch_loss
                benchmark_data[block_name]["epoch_loss"] = epoch_loss
                
        # Single GPU Training
        if config["single_gpu"] and has_gpus:
            logging.info("Single GPU Training")
            block_name = "gpu_{0:02d}_training".format(1)
            start_timing(benchmark_data, block_name, parent_p, out_path)
            epoch_times, batch_loss, epoch_loss = train_conv_net_gpu(train_data_scaled, train_counts,
                            val_data_scaled, val_counts, config["conv_net_parameters"],
                            1, config["random_seed"], dtype=config["dtype"])
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
        parent_p.send("exit")
        monitor_proc.join()
        sys.exit()
    return


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

if __name__ == "__main__":
    main()
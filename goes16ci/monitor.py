from multiprocessing import Process, Pipe
import logging
import os
import numpy as np
import pandas as pd
import pickle
import psutil
import re
import sys
import time
from time import perf_counter, process_time
from subprocess import Popen, PIPE
from io import StringIO
from shutil import which
from os.path import exists, join


class Monitor(object):
    """Monitor memory and other aspects of a process. 
    """
    def __init__(self, p, sleep_interval=0.2):
        self.pipe = p
        self.sleep_interval = sleep_interval
        self.monitoring_output_path = ""
        self.proc = None
        self.time_values = list()
        self.monitor_values = dict()
        self.monitor_values["cpu_memory_percent"] = list()
        self.monitor_values["cpu_util_percent"] = list()
        self.monitor_output_path = "/tmp/monitor_out.csv"
        self.gpu = False
        self.num_gpus = 0
        self.gpu_names = []
        if which("nvidia-smi") is not None:
            self.gpu = True
            self.gpu_names = get_gpu_names()
            self.num_gpus = len(self.gpu_names)
            for gpu_num in range(self.num_gpus):
                self.monitor_values[f"gpu_util_percent_{gpu_num:d}"] = list()
                self.monitor_values[f"gpu_memory_total_MiB_{gpu_num:d}"] = list()
                self.monitor_values[f"gpu_memory_used_MiB_{gpu_num:d}"] = list()
                self.monitor_values[f"gpu_memory_used_percent_{gpu_num:d}"] = list()

    def run(self):
        if hasattr(os, 'getppid'):
            ppid = os.getppid()
            self.proc = psutil.Process(ppid)

        else:
            logging.info("Monitor::run: no getppid function!")
            sys.exit(1)

        running = True
        while running:
            msg = self.pipe.recv()
            if re.match("start", msg) is not None:
                self.monitor_output_path = msg.split()[1]
                self.mem_values = list()
                self.monitor_process()

            elif re.match("exit", msg) is not None:
                running = False

            else:
                logging.info("Monitor::monitor_process: " + \
                             "unrecognized msg {0}".format(msg))
                sys.exit(1)

    def monitor_process(self):
        monitoring_process = True
        while monitoring_process:
            self.time_values.append(perf_counter())
            self.monitor_values["cpu_memory_percent"].append(self.proc.memory_percent(memtype="vms"))
            self.monitor_values["cpu_util_percent"].append(self.proc.cpu_percent(interval=0.0))
            if self.gpu:
                gpu_stats = get_gpu_util_stats()
                for gpu_num in range(self.num_gpus):
                    self.monitor_values[f"gpu_util_percent_{gpu_num:d}"].append(gpu_stats.iloc[gpu_num, 0])
                    self.monitor_values[f"gpu_memory_total_MiB_{gpu_num:d}"].append(gpu_stats.iloc[gpu_num, 1])
                    self.monitor_values[f"gpu_memory_used_MiB_{gpu_num:d}"].append(gpu_stats.iloc[gpu_num, 2])
                    self.monitor_values[f"gpu_memory_used_percent_{gpu_num:d}"].append(100 * gpu_stats.iloc[gpu_num, 2] / gpu_stats.iloc[gpu_num, 1])
            time.sleep(self.sleep_interval)
           
            # check for a stop message
            if self.pipe.poll():
                msg = self.pipe.recv()
                if msg == "stop":
                    #
                    # stop message received; write data to file
                    #
                    monitoring_process = False
                    monitor_values_df = pd.DataFrame(index=self.time_values, data=self.monitor_values)
                    monitor_values_df.to_csv(self.monitor_output_path, index_label="time")
                    del self.time_values[:]
                    for mvk in self.monitor_values.keys():
                        del self.monitor_values[mvk][:]
                    self.pipe.send("saved")
                else:
                    logging.info("Monitor::monitor_process: " +
                                 "unrecognized msg {0}".format(msg))
                    sys.exit(1)

        return


def get_cuda_version():
    if which("nvidia-smi") is None:
        gpu_version_info = {"cuda_version": "-1", "gpu_driver_version": "-1"}
    else:
        proc = Popen(["nvidia-smi"], stdout=PIPE)
        stdout, stderr = proc.communicate()
        gpu_version_str = stdout.decode("UTF-8").split("\n")[2].split()
        gpu_version_info = {"cuda_version": gpu_version_str[-2],
                        "gpu_driver_version": gpu_version_str[-5]}
    return gpu_version_info


def get_cudnn_version():
    nvcc_path = which("nvcc")
    if nvcc_path is None:
        cudnn_version_info = "-1"
    else:
        cuda_top_path = "/".join(nvcc_path.split("/")[:-2])
        proc = Popen(["find", cuda_top_path, "-name", "*cudnn.so.*"], stdout=PIPE)
        stdout, stderr = proc.communicate()
        cudnn_version_info = stdout.decode("utf-8").strip().split("\n")[-1].split("/")[-1].strip("libcudnn.so.")
    return cudnn_version_info

def get_nccl_version():
    nvcc_path = which("nvcc")
    if nvcc_path is None:
        nccl_version_info = "-1"
    else:
        cuda_top_path = "/".join(nvcc_path.split("/")[:-2])
        proc = Popen(["find", cuda_top_path, "-name", "*nccl.so.*"], stdout=PIPE)
        stdout, stderr = proc.communicate()
        nccl_version_info = stdout.decode("utf-8").strip().split("\n")[-1].split("/")[-1].strip("libnccl.so.")
    return nccl_version_info
    

def get_gpu_names():
    """
    Get the names for each GPU on the system.

    Returns:

    """
    if which("nvidia-smi") is None:
        gpu_name_list = []
    else:
        proc = Popen(["nvidia-smi", "-L"], stdout=PIPE)
        stdout, stderr = proc.communicate()
        gpu_name_str = stdout.decode("UTF-8")
        gpu_name_list = gpu_name_str.strip().split("\n")
    return gpu_name_list



def get_gpu_topo():
    """
    Get the names for each GPU on the system.

    Returns:

    """
    if which("nvidia-smi") is None:
        gpu_topo_str = ""
    else:
        proc = Popen(["nvidia-smi", "topo", "-m"], stdout=PIPE)
        stdout, stderr = proc.communicate()
        gpu_topo_str = stdout.decode("UTF-8")
    return gpu_topo_str


def get_gpu_util_stats():
    """
    Gets GPU usage statistics using the nvidia-smi command.

    Returns:
        pandas DataFrame containing usage stats for each GPU.
    """
    proc = Popen(["nvidia-smi",
                  "--query-gpu=index,utilization.gpu,memory.total,memory.used",
                  "--format=csv,nounits"], stdout=PIPE)
    stdout, stderr = proc.communicate()
    gpu_stat_str = stdout.decode("UTF-8")
    return pd.read_csv(StringIO(gpu_stat_str), index_col="index")


def start_timing(benchmark_data, block_name, monitor_pipe, out_path):
    """
    Start the timing for a block of code.

    Args:
        benchmark_data: dictionary of benchmark attributes
        block_name: Name of the block being timed.
        monitor_pipe: 
    """
    benchmark_data[block_name] = {}
    benchmark_data[block_name]["elapsed_start"] = perf_counter()
    benchmark_data[block_name]["process_start"] = process_time()
    monitor_pipe.send("start " + join(out_path, block_name + "_stats.csv"))


def end_timing(benchmark_data, epoch_times, block_name, monitor_pipe, out_path):
    benchmark_data[block_name]["elapsed_end"] = perf_counter()
    benchmark_data[block_name]["process_end"] = process_time()

    benchmark_data[block_name]["elapsed_duration"] = benchmark_data[block_name]["elapsed_end"] - \
        benchmark_data[block_name]["elapsed_start"]
    benchmark_data[block_name]["process_duration"] = benchmark_data[block_name]["process_end"] - \
        benchmark_data[block_name]["process_start"]
    benchmark_data[block_name]["epoch_duration"] = epoch_times[-1]
    monitor_pipe.send("stop")
    monitor_pipe.recv()
    calc_summary_stats(benchmark_data, block_name, join(out_path, block_name + "_stats.csv"))

def calc_summary_stats(benchmark_data, block_name, output_file):
    if not exists(output_file):
        raise FileNotFoundError(output_file + " does not exist")
    stats = pd.read_csv(output_file, index_col="time")
    for col in stats.columns:
        print(col)
        benchmark_data[block_name][col + "_max"] = float(stats[col].max())
        benchmark_data[block_name][col + "_min"] = float(stats[col].min())
        benchmark_data[block_name][col + "_mean"] = float(stats[col].mean())
        benchmark_data[block_name][col + "_median"] = float(stats[col].median())


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level="INFO")
    (parent_p, child_p) = Pipe()
    m = Monitor(child_p)
    p = Process(target=m.run)
    print("Starting process")
    p.start()
    parent_p.send("start /tmp/monitor_train.txt")
    time.sleep(5)
    parent_p.send("stop")
    print("stopping process")
    parent_p.send("start /tmp/monitor_predict.txt")
    time.sleep(5)
    parent_p.send("stop")
    parent_p.send("exit")

    p.join()


#!/bin/bash -l
#PBS -N goes16ci
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=8:ngpus=1:mem=384GB
#PBS -A NAML0001
#PBS -q casper
#PBS -l gpu_type=v100
### Merge output and error files
#PBS -j oe
#PBS -k eod
export PATH=/glade/u/home/gwallach/.conda/envs/goes/bin:$PATH
#python -u process_goes16.py goes16_process_config.yml -n 36 -l
python -u process_goes16.py goes16_process_config.yml -n 42 -a >&process.log
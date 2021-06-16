#!/bin/bash -l
#PBS -N goes16ci
#PBS -A NAML0001
#PBS -l -walltime=05:00:00
#PBS -q casper
#PBS -l select=N:mem=128G
export PATH=/glade/u/home/gwallach/.conda/envs/goes/bin:$PATH
module load cuda/11 cudnn nccl
python -u goes16_deep_learning_benchmark.py -c benchmark_config_default-Gunther.yml >& goes_deep_default.log

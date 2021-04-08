#!/bin/bash -l
#PBS -N goes16ci
#PBS -A NAML0001
#PBS -l walltime=01:00:00
#PBS -j oe
#PBS -q regular
#PBS -l select=1:ncpus=36
#PBS -m abe
#PBS -M dgagne@ucar.edu
export PATH=/glade/u/home/gwallach/.conda/envs/goes/bin:$PATH
module load cuda/11 cudnn nccl
python -u goes16_deep_learning_benchmark.py >& goes_deep_chey.log

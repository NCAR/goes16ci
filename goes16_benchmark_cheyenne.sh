#!/bin/bash -l
#PBS -N goes16ci
#PBS -A NAML0001
#PBS -l walltime=01:00:00
#PBS -j oe
#PBS -q regular
#PBS -l select=1:ncpus=36
#PBS -m abe
#PBS -M dgagne@ucar.edu
export PATH=/glade/u/home/dgagne/miniconda3/envs/goes/bin:$PATH
python -u goes16_deep_learning_benchmark.py -c benchmark_config_cheyenne.yml >& goes_deep_chey_2.log

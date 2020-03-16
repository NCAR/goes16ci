#!/bin/bash -l
#PBS -N goes16ci
#PBS -A NAML0001
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -q regular
#PBS -l select=1:ncpus=36
#PBS -m abe
#PBS -M dgagne@ucar.edu
#export PATH=/glade/u/home/dgagne/miniconda3/envs/ml/bin:$PATH
cd ~/goes16ci
module load ncarenv/1.3   python/3.7.5   gnu/7.4.0   ncarcompilers/0.5.0   netcdf/4.7.3 mpt/2.19
export PATH="/glade/work/dgagne/chey_20191211/bin/python:$PATH"
python -u goes16_deep_learning_benchmark.py -c benchmark_config_cheyenne.yml >& goes_deep_chey.log

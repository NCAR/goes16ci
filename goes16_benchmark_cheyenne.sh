#!/bin/bash -l
#PBS -N goes16ci
#PBS -A NAML0001
#PBS -l walltime=01:00:00
#PBS -j oe
#PBS -q regular
#PBS -l select=1:ncpus=36
#PBS -m abe
#PBS -M dgagne@ucar.edu
module purge
export PATH=/glade/u/home/gwallach/miniconda3/envs/ml/bin:$PATH
cd ~/goes16ci
python setup.py install
python -u goes16_deep_learning_benchmark.py >& goes_deep_chey.log

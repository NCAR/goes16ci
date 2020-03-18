#!/bin/bash -l
#SBATCH --job-name=goes16ci
#SBATCH --account=NAML0001
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --time=06:00:00
#SBATCH --partition=dav
#SBATCH --gres=gpu:v100:8
#SBATCH --exclusive
#SBATCH --mem=256G
module purge
module load gnu/7.3.0 openmpi python/3.6.8 cuda/10.0 #changed to python 3.6 instead of 3.7 01/16 because tf2.0 is only compatable with python 3.6
export PATH="/glade/u/home/gwallach/.conda/envs/goes16/bin:$PATH"
module load ncarenv
ncar_pylib
python setup.py install
python -u goes16_deep_learning_benchmark_32.py >& goes_deep32.log
#python -u goes16_deep_learning_benchmark_64.py >& goes_deep64.log
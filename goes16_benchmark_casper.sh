#!/bin/bash -l
#SBATCH --job-name=goes16ci
#SBATCH --account=NAML0001
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --partition=dav
#SBATCH --gres=gpu:v100:4
#SBATCH --exclusive
#SBATCH --mem=128G
module purge
module load gnu/7.3.0 openmpi-x/3.1.0 python/3.6.4 cuda
source /glade/work/dgagne/ncar_pylib_dl/bin/activate
cd ~/goes16ci
python setup.py install
python -u goes16_deep_learning_benchmark.py >& goes_deep.log

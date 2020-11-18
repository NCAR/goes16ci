#!/bin/bash -l
#SBATCH --job-name=goes16ci
#SBATCH --account=NAML0001
#SBATCH --ntasks=32
#SBATCH --time=04:00:00
#SBATCH --partition=dav
#SBATCH --mem=128G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:v100:1
module purge
source activate goes
python -u goes16_deep_learning_benchmark.py -c benchmark_config_default >& goes_deep.log

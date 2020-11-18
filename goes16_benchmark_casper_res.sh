#!/bin/bash -l
#SBATCH --job-name=goes16ci
#SBATCH --account=NAML0001
#SBATCH --ntasks=36
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --partition=dav
#SBATCH --gres=gpu:v100:8
#SBATCH --exclusive
#SBATCH --reservation=casper_8xV100
#SBATCH --mem=768G
module purge
source activate goes
python -u goes16_deep_learning_benchmark.py -c benchmark_config_default.yml >& goes_deep_default.log

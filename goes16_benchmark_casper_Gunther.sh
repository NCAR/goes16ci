#!/bin/bash -l
#SBATCH --job-name=goes16ci
#SBATCH --account=NAML0001
#SBATCH --ntasks=32
#SBATCH --time=01:00:00
#SBATCH --partition=dav
#SBATCH --mem=128G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:v100:1
module purge
module load ncarenv/1.3 gnu/7.4.0 ncarcompilers/0.5.0 python/3.7.5 cuda/10.1
ncar_pylib  
pip install -e .
python -u goes16_deep_learning_benchmark.py -c benchmark_config_default_Gunther.yml >& goes_deep32.log
#python -u goes16_deep_learning_benchmark_64.py >& goes_deep64.log


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
module load ncarenv/1.3 gnu/7.4.0 ncarcompilers/0.5.0 python/3.7.5 cuda/10.0
ncar_pylib  
pip install -e .
python -u goes16_deep_learning_benchmark.py -c benchmark_config_default_Gunther.yml >& goes_deep32.log
#python -u goes16_deep_learning_benchmark_64.py >& goes_deep64.log
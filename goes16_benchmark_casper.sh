#!/bin/bash -l
#SBATCH --job-name=goes16ci
#SBATCH --account=NAML0001
#SBATCH --ntasks=36
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --partition=dav
#SBATCH --gres=gpu:v100:8
#SBATCH --exclusive
#SBATCH --mem=768G
module load ncarenv/1.3 gnu/7.4.0 ncarcompilers/0.5.0 python/3.7.5 cuda/10.1
ncar_pylib ncar_20191211 
pip install -e .
python -u goes16_deep_learning_benchmark.py -c benchmark_config_default.yml >& goes_deep_default.log

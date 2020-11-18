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
#module load ncarenv/1.3 gnu/7.4.0 ncarcompilers/0.5.0 python/3.7.5 cuda/10.0
#ncar_pylib ncar_20191211 
module load cuda/10.1
export PATH="/glade/u/home/dgagne/miniconda3/envs/goes/bin/:$PATH"
pip install -e .
python -u goes16_benchmark.py -c benchmark_config_default-Gunther.yml >& goes_deep_default.log

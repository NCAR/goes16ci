#!/bin/bash -l
#SBATCH --account=NAML0001
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=768G
#SBATCH -n 8
#SBATCH -t 24:00:00
#SBATCH -J goes_hyper
#SBATCH -o goes_hyper.out
#SBATCH -e goes_hyper.err
module load ncarenv/1.3 gnu/8.3.0 openmpi/3.1.4 python/3.7.5 cuda/10.1
ncar_pylib /glade/work/schreck/py37
python /glade/work/schreck/py37/lib/python3.7/site-packages/aimlutils/hyper_opt/run.py hyperparameter.yml benchmark_config_default-Gunther.yml

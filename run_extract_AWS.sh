#!/bin/bash -l
#SBATCH --job-name=goes16ci
#SBATCH --ntasks=36
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:v100:1
#SBATCH --account=NAML0001
#SBATCH --time=24:00:00
#SBATCH --mem=156G
module load ncarenv
ncar_pylib
source activate goes
python scripts/extract_goes16_AWS.py >extract.log
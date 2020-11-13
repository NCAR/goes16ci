#!/bin/bash -l
#SBATCH --job-name=goes16_process
#SBATCH --account=NAML0001
#SBATCH --time=12:00:00
#SBATCH --partition=dav
#SBATCH --ntasks=36
#SBATCH --ntasks-per-node=36
#SBATCH -e goes16_process.log
#SBATCH -o goes16_process.log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=dgagne
module purge
export PATH="/glade/u/home/gwallach/.conda/envs/goes16/bin:$PATH"
cd $HOME/goes16ci/
pip install .
cd scripts
python -u process_goes16.py goes16_process_config.yml -n 36 -a 

#!/bin/bash -l
#SBATCH -J goes16_hpss
#SBATCH -A NAML0001
#SBATCH -t 24:00:00
#SBATCH -p hpss
#SBATCH -n 1
#SBATCH --ntasks-per-node=5
#SBATCH -e goes16_hpss.log
#SBATCH -o goes16_hpss.log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=dgagne
source ~/.bash_profile
export PATH="/glade/u/home/dgagne/miniconda3/envs/deep/bin:$PATH"
cd ~/goes16ci/scripts
python -u untar_goes16.py >& untar.log
#python -u extract_goes16_hpss.py -o /glade/scratch/dgagne/goes16_nc/ -d 2018-04-01 -e 2018-06-06 -n 5 >& extract.log
#python -u extract_goes16_hpss.py -o /glade/work/dgagne/goes16_nc/ -i ABI-L1b -s conus -d 2018-04-01 -e 2018-06-06 -n 5 >& extract_conus.log

#!/bin/bash -l
#SBATCH -J goes16_hpss
#SBATCH -A NAML0001
#SBATCH -t 24:00:00
#SBATCH -p hpss
#SBATCH -n 1
#SBATCH --ntasks-per-node=5
#SBATCH -e goes16_hpss.log
#SBATCH -o goes16_hpss.log
export PATH="/glade/u/home/dgagne/miniconda3/envs/tf2/bin:$PATH"
echo `which python`
echo $path
cd ~/goes16ci/scripts
python -u extract_goes16_hpss.py -o /glade/scratch/$USER/goes16_nc/ -d 2019-03-01 -e 2019-05-01 -n 5 >& extract.log
python -u extract_goes16_hpss.py -o /glade/scratch/$USER/goes16_nc/ -i ABI-L1b -s conus -d 2019-03-01 -e 2019-05-01 -n 5 >& extract.log
python -u untar_goes16.py >& extract.log


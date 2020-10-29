#!/bin/bash -l
#SBATCH -J goes16_hpss
#SBATCH -A NAML0001
#SBATCH -t 24:00:00
#SBATCH -p dav
#SBATCH -n 1
#SBATCH --ntasks-per-node=5
#SBATCH -e goes16_hpss.log
#SBATCH -o goes16_hpss.log
module purge
module load ncarenv/1.3 gnu/7.4.0 ncarcompilers/0.5.0 python/3.7.5 cuda/10.1
ncar_pylib  
pip install -e .
cd ~/goes16ci/scripts
<<<<<<< HEAD
#python -u extract_goes16_hpss.py -o /glade/scratch/$USER/goes16/ -d 2019-03-01 -e 2019-10-01 -n 5 >& extract.log
#python -u extract_goes16_hpss.py -o /glade/scratch/$USER/goes16/ -i ABI-L1b -s conus -d 2019-08-01 -e 2019-10-01 -n 5 >& extract.log
python -u untar_goes16.py >& untar_goes16.log
=======
python -u extract_goes16_hpss.py -o /glade/scratch/$USER/goes16_nc/ -d 2019-03-01 -e 2019-05-01 -n 5 >& extract.log
python -u extract_goes16_hpss.py -o /glade/scratch/$USER/goes16_nc/ -i ABI-L1b -s conus -d 2019-03-01 -e 2019-05-01 -n 5 >& extract.log
python -u untar_goes16.py >& extract.log

>>>>>>> 8e3b2e64068e40c4a4b42a8c38663ad4a8b4ce43

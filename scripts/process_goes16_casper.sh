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
<<<<<<< HEAD
#SBATCH --mail-user=gwallach
export PATH="/glade/u/home/gwallach/.conda/envs/goes16/bin:$PATH"
cd $HOME/goes16ci/
pip install .
cd scripts
python -u process_goes16.py goes16_process_config_32.yml -n 6 -l >& process.log
python -u process_goes16.py goes16_process_config_64.yml -n 6 -l >& process.log
python -u process_goes16.py goes16_process_config_32.yml -n 6 -a >& process.log
python -u process_goes16.py goes16_process_config_64.yml -n 6 -a >& process.log
#python -u process_goes16.py goes16_process_config_128.yml -n 36 -a >& process.log
=======
#SBATCH --mail-user=dgagne
module purge
export HOME="/glade/u/home/gwallach"
module load gnu/7.3.0 openmpi-x/3.1.0 python/3.6.4 cuda netcdf
source /glade/work/dgagne/ncar_pylib_dl/bin/activate
cd $HOME/goes16ci
python setup.py install
cd $HOME/goes16ci/scripts
python -u process_goes16.py goes16_process_config.yml -n 36 -a 
>>>>>>> 8e3b2e64068e40c4a4b42a8c38663ad4a8b4ce43

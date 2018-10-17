#!/bin/bash -l
#PBS -N goes16_hpss
#PBS -A NAML0001
#PBS -l walltime=06:00:00
#PBS -q economy
#PBS -j oe
#PBS -m abe
#PBS -M dgagne@ucar.edu
#PBS -l select=1:ncpus=6:mpiprocs=6
module purge
source ~/.bash_profile
export PATH="/glade/u/home/dgagne/miniconda3/envs/deep/bin:$PATH"
cd ~/goes16ci/scripts
python -u extract_goes16_hpss.py -o /glade/scratch/dgagne/goes16_nc/ -d 2018-04-01 -e 2018-06-30 -n 5 >& extract.log 

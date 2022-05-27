#!/bin/bash -l
#PBS -N goes16ci
#PBS -A NAML0001
#PBS -l walltime=03:00:00
#PBS -l select=1:ncpus=8:mem=320GB:ngpus=0
#PBS -q casper
### Merge output and error files
#PBS -j oe
#PBS -k eod
export PATH=/glade/u/home/pabharathi/miniconda/envs/myenv/bin:$PATH
python -u process_goes16-UNET.py goes16_process_config.yml -n 42 -a >&process.log
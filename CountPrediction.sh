#!/bin/bash -l
#PBS -N jupyter_instance
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=8:ngpus=1:mem=128GB
#PBS -A NAML0001
#PBS -q casper
#PBS -l gpu_type=v100
### Merge output and error files
#PBS -j oe
#PBS -k eod
export PATH=/glade/u/home/gwallach/.conda/envs/goes/bin:$PATH
module load cuda/11 cudnn nccl
python scripts/BinnedCountPredictionModel.py >& countprediction.log
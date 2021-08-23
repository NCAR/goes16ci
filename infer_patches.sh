#!/bin/bash -l
#PBS -N goes16ci
#PBS -A NAML0001
#PBS -l walltime=05:00:00
#PBS -q casper
#PBS -l select=1:mem=128G
export PATH=/glade/u/home/gwallach/.conda/envs/goes/bin:$PATH
module load cuda/11 cudnn nccl
python patch_inference.py -v "/glade/u/home/gwallach/goes16ci/goes16_resnet_gpus_01.h5" /glade/scratch/gwallach/goes16_nc/abi_patches_20190501T070000.nc /glade/scratch/gwallach/goes16_nc/glm_grid_s20190501T070000_e20190501T234000.nc lightning_20190501T070000_hyperop_out.nc >& patches.log
#!/bin/bash -l
#PBS -N goes16ci
#PBS -A NAML0001
#PBS -l walltime=04:00:00
#PBS -q casper
export PATH=/glade/u/home/gwallach/.conda/envs/goes/bin:$PATH
python GLM_LightningCount.py >& lightningcount.log
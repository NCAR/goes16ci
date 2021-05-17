#!/bin/bash -l
#PBS -N goes16ci
#PBS -A NAML0001
#PBS -l -walltime=00:30:00
#PBS -q casper
#PBS -l select=N:mem=128G
export PATH=/glade/u/home/gwallach/.conda/envs/goes/bin:$PATH
python prediction_vs_actual_visual.py "/glade/u/home/gwallach/goes16ci/lightning_20190501T070000_hyperop_out.nc" "/glade/scratch/gwallach/goes16_nc/glm_grid_s20190501T070000_e20190501T234000.nc" > visual.log
convert PredvsActual2019-05-01T07:20:00-2019-05-01T23:40:00.gif -coalesce -repage 0x0 -trim +repage 20190501T07-24_class_weight.gif
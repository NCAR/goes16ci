#!/bin/bash -l
#SBATCH --job-name=goes16ci
#SBATCH --account=NAML0001
#SBATCH --time=12:00:00
#SBATCH --partition=dav
#SBATCH --mem=128G
module load ncarenv
ncar_pylib
source activate goes
python patch_inference.py -v "/glade/u/home/gwallach/goes16ci/goes16ci_model_cpu2020-11-10 14:28:04.352884.h5" /glade/scratch/gwallach/goes16_nc/abi_patches_20190501T030000.nc /glade/scratch/gwallach/goes16_nc/glm_grid_s20190501T030000_e20190501T234000.nc lightning_20190501T030000_out.nc >& patches.log
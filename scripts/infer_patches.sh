#!/bin/bash -l
#SBATCH --job-name=goes16ci
#SBATCH --account=NAML0001
#SBATCH --time=12:00:00
#SBATCH --partition=dav
#SBATCH --mem=128G
#module load ncarenv/1.3 gnu/7.4.0 ncarcompilers/0.5.0 python/3.7.5 cuda/10.0
#ncar_pylib ncar_20191211 
module load cuda/10.1
export PATH="/glade/u/home/dgagne/miniconda3/envs/goes/bin/:$PATH"
pip install -e 
python patch_inference.py -v "/glade/u/home/gwallach/goes16ci/goes16ci_model_cpu2020-09-16 21:38:55.333504.h5" /glade/scratch/gwallach/goes16_nc/ABI_patches_32/abi_patches_20190901T000000.nc /glade/scratch/gwallach/goes16_nc/glm_grid_s20190901T180000_e20190901T230000.nc lightning_20190901T18000_out.nc >& patches.log
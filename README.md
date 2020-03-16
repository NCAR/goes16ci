# GOES 16 Lightning Count Prediction Benchmark

[![Build Status](https://travis-ci.com/NCAR/goes16ci.svg?branch=master)](https://travis-ci.com/NCAR/goes16ci)

The GOES 16 Lightning Count Prediction benchmark is a deep learning benchmark for HPC systems 
used for atmospheric science problems. 

## Contributors
* David John Gagne
* Bill Anderson
* Gunther Wallach

## Requirements
The code is designed to run on Python 3.6 and 3.7. It requires the following
Python libraries:
* numpy
* scipy
* pandas
* xarray
* tensorflow>=2.0.0
* scikit-learn
* pyproj
* dask distributed (for data processing)
* ipython 
* jupyter (for interactive visualization of neural networks)

## Setup from Scratch

* Install Python 3.7 on your machine. I recommend the Miniconda Python installer available
[here](https://docs.conda.io/en/latest/miniconda.html).

* Create a benchmark environment: `conda create -n goes16 python=3.7`

* Once the environment is installed activate it on your machine:
`source activate goes16`

* Install the needed Python libraries from conda

```bash
conda install -c conda-forge --yes \
    pip \
    ipython \
    jupyter \
    numpy \
    scipy \
    matplotlib \
    xarray \
    netcdf4 \
    pandas \
    pyyaml \
    dask \
    distributed \
    scikit-learn \
    pyproj
```

* Make sure the CUDA kernel and CUDA toolkit are installed on your system and know the path
and versions. 

* Install the tensorflow-gpu binary (if installing tensorflow 1.15) or tensorflow binary (if tensorflow 2). For more detailed installation instructions 
visit the [tensorflow website](https://www.tensorflow.org/install/gpu).
```bash
# If you plan to use tensorflow 2
pip install tensorflow
```
## Run Benchmark Script

* Clone the goes16ci git repository to your home directory.
```bash
cd ~
git clone https://github.com/NCAR/goes16ci.git
cd goes16ci
```

* Install the goes16ci library
```bash
pip install .
```

* Download the GOES16 patch files. You will need about 8 GB free to download 
and untar the data.
```bash
python download_data.py
```
* Run the benchmark script. The script will output trained neural networks and a yaml file
with the timing information for each step.
```bash
python goes16_deep_learning_benchmark.py
```

* If you want to modify the neural network or other properties of the script,
you can make a copy of benchmark_config_default.yml and modify it. To run the script with the
new config file:
```bash
python goes16_deep_learning_benchmark.py -c benchmark_config_default.yml
```

## Setup on Cheyenne/Casper

* Clone the git repo to your home directory
```bash
cd ~
git clone https://github.com/NCAR/goes16ci.git
cd goes16ci
```

* Create a link to the patch data on GLADE
```bash
ln -s /glade/p/cisl/aiml/dgagne/goes16_nc/ABI_patches_20190315 data
```

* Modify the `goes16_benchmark_casper.sh` script with your account number.

* Submit the benchmark script to the casper queue:
`sbatch goes16_benchmark_casper.sh`

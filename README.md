# GOES 16 Lightning Count Prediction Benchmark

The GOES 16 Lightning Count Prediction benchmark is a deep learning benchmark for HPC systems 
used for atmospheric science problems. 

## Requirements
The code is designed to run on Python 3.6. It requires the following
Python libraries:
* numpy
* scipy
* pandas
* xarray
* tensorflow
* keras
* scikit-learn
* pyproj
* dask distributed (for data processing)
* ipython 
* jupyter (for interactive visualization of neural networks)

## Setup from Scratch

* Install Python 3.6 on your machine. I recommend the Miniconda Python installer available
[here](https://docs.conda.io/en/latest/miniconda.html).

* Create a benchmark environment: `conda create -n goes16 python=3.6`

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
and versions. The tensorflow 1.13 binary is built with CUDA 10 while tensorflow 1.12 is built with CUDA 9.
Use 1.12 or 1.13 depending on which version your system has available or build tensorflow from source
to maximize performance.

* Install the tensorflow-gpu binary and keras. For more detailed installation instructions 
visit the [tensorflow website](https://www.tensorflow.org/install/gpu).
```bash
# If you have CUDA 10 installed
pip install "tensorflow-gpu==1.13"
# If you have CUDA 9 installed
pip install "tensorflow-gpu==1.12"
# Install keras
pip install keras
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
python setup.py install
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
you can make a copy of benchmark_config.yml and modify it. To run the script with the
new config file:
```bash
python goes16_deep_learning_benchmark.py -c config_file.yml
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
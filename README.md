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
from urllib.request import urlretrieve
import os
from os.path import exists, join
import tarfile
if not exists("data"):
    os.mkdir("data")
nc_tar_file = "https://storage.googleapis.com/goes16ci_deep_learning_benchmark_data/ABI_patches_20190315.tar.gz"
print("Get nc files")
urlretrieve(nc_tar_file, join("data", nc_tar_file.split("/")[-1]))
print("Extract csv tar file")
print("Extract nc tar file")
nc_tar = tarfile.open(join("data", nc_tar_file.split("/")[-1]))
nc_tar.extractall("data/")
nc_tar.close()

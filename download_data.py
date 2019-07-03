from urllib.request import urlretrieve
from os.path import join
from shutil import move
import tarfile
import os

nc_tar_file = "https://storage.googleapis.com/goes16ci_deep_learning_benchmark_data/ABI_patches_20190315.tar.gz"
print("Get nc files")
urlretrieve(nc_tar_file, join(nc_tar_file.split("/")[-1]))
print("Extract csv tar file")
print("Extract nc tar file")
nc_tar = tarfile.open(join(nc_tar_file.split("/")[-1]))
nc_tar.extractall("./")
nc_tar.close()
move("./ABI_patches_20190315", "./data/")
os.remove("ABI_patches_20190315.tar.gz")


from goes16ci.data import download_data
from os.path import join
import os
import s3fs as s3

download_data('2019-12-04','2019-12-31','ABI-L1b','RadC','/glade/p/cisl/aiml/gwallach/goes16_nc/')
#download_data('2019-11-27','2019-12-31','ABI-L1b','RadC','/glade/scratch/gwallach/goes16_nc/')

#download_data('2018-04-11','2018-12-31','GLM-L2','LCFA','/glade/scratch/gwallach/goes16_nc/')
#download_data('2019-01-01','2019-03-01','GLM-L2','LCFA','/glade/scratch/gwallach/goes16_nc/')
#download_data('2019-12-01','2019-12-31','GLM-L2','LCFA','/glade/scratch/gwallach/goes16_nc/')
#download_data('2019-06-15','2019-12-31','GLM-L2','LCFA','/glade/scratch/gwallach/goes16_nc/')

from glob import glob
from os import listdir, mkdir
from os.path import join, exists
import subprocess

def main():
    #print("Within Tar Main File")
    tar_path = "/glade/scratch/gwallach/goes16/"
    out_path = "/glade/scratch/gwallach/goes16/goes16_nc/"
    instruments = sorted(listdir(tar_path))
    for instrument in instruments:
        #print("Instrument's Loop")
        if not exists(join(out_path, instrument)):
            mkdir(join(out_path, instrument))
        dates = sorted(listdir(join(tar_path, instrument)))
        for date in dates:
            print("Dates Loop")
            print(instrument, date)
            if not exists(join(out_path, instrument, date)):
                mkdir(join(out_path, instrument, date))
            print("In if not exists")
            tar_files = sorted(glob(join(tar_path, instrument, date, "*.tar")))
            print("past line 22")
            for tar_file in tar_files:
                print("tar_file loop")
                tar_command = ['tar', '-xvf',tar_file, join(out_path, instrument, date)]
                j_join = join(out_path, instrument, date)
                print(tar_command)
                subprocess.call(['tar','-xvf',tar_file,'--strip-components=3', '-C', j_join])
                
if __name__ == "__main__":
    main()

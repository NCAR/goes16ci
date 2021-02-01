from glob import glob
from os import listdir, mkdir, makedirs
from os.path import join, exists
import subprocess

def main():
    tar_path = "/glade/scratch/gwallach/goes16_nc/"
    out_path = "/glade/p/cisl/aiml/gwallach/goes16_nc/"
    instruments = sorted(listdir(tar_path))
    for instrument in instruments:
        #print("Instrument's Loop")
        dates = sorted(listdir(join(tar_path, instrument)))
        for date in dates:
            #print("Dates Loop")
            print(instrument, date)
            if not exists(join(out_path, instrument, date)):
                makedirs(join(out_path, instrument, date))
            #print("In if not exists")
            tar_files = sorted(glob(join(tar_path, instrument, date, "*.tar")))
            #print("past line 22")
            for tar_file in tar_files:
                #print("tar_file =",tar_file)
                j_join = join(out_path, instrument, date)
                tar_command = ['tar','-xvf',tar_file,'-C',j_join]
                #tar_command = ['tar -xvf' {tar_file} '--strip-components=3 -C' {j_join}]
                #tar_command = f"tar -xvf {tar_file} --strip-components=3 -C {join(out_path, instrument, date)}"
                #print(tar_command)
                #subprocess.call(tar_command,shell = True)
                #subprocess.Popen('tar','-xvf',tar_file,'--strip-components=3','-C', j_join)
                subprocess.call(tar_command)
                
if __name__ == "__main__":
    main()

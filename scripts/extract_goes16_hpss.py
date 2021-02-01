import subprocess
from os import makedirs, chdir
from os.path import exists, join
import argparse
import traceback
from multiprocessing import Pool
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", default="/FS/EOL/operational/satellite/goes/g16/", 
                        help="Path to HPSS top level GOES-16 directory")
    parser.add_argument("-t", "--tout", default="/glade/scratch/gwallach/goes16_nc/", help="Path to tar files")
    parser.add_argument("-o", "--out", default="/glade/scratch/gwallach/goes16_nc/", help="Path where output netCDF files are extracted")
    parser.add_argument("-i", "--ins", default="GLM-L2", choices=["ABI-L1b", "GLM-L2"], 
                        help="Instrument on satellite")
    parser.add_argument("-s", "--sec", default="LCFA", choices=["LCFA", "conus", "fdisk", "meso"], help="Sector")
    parser.add_argument("-d", "--startdate", help="Start date of extraction")
    parser.add_argument("-e", "--enddate", help="Last date of extraction (inclusive)")
    parser.add_argument("-n", "--nproc", type=int, default=1, help="Number of processes")
    args = parser.parse_args()
    dates = pd.date_range(start=args.startdate, end=args.enddate, freq="1D")
    if args.nproc > 1:
        pool = Pool(args.nproc)
        for date in dates:
            pool.apply_async(extract_hpss_tar_file, (date, args.ins, args.sec, args.path, args.tout))
        pool.close()
        pool.join()
    else:
        for date in dates:
            extract_hpss_tar_file(date, args.ins, args.sec, args.path, args.tout)
    return

def extract_hpss_tar_file(date, instrument, sector, hpss_path, tar_out_path):
    try:
        year = date.year
        jday = date.dayofyear
        date_str = date.strftime("%Y%m%d")
        if instrument == "GLM-L2":
            tar_filename = join(hpss_path, f"{year}", f"day{jday:03}", f"OR_{instrument}_g16_{sector}_{date_str}.tar")
            if not exists(tar_out_path):
                makedirs(tar_out_path)
            full_tar_out_path = join(tar_out_path, instrument + "_" + sector, date_str)
            if not exists(full_tar_out_path):
                makedirs(full_tar_out_path)
            chdir(full_tar_out_path)
            subprocess.call(["hsi", "get", tar_filename])
        else:
            if sector == "fdisk":
                ending = "00.tar"
            else:
                ending = ".tar"
            if not exists(tar_out_path):
                makedirs(tar_out_path)
            full_tar_out_path = join(tar_out_path, instrument + "_" + sector, date_str)
            if not exists(full_tar_out_path):
                makedirs(full_tar_out_path)
            chdir(full_tar_out_path)
            for hour in range(24):
                tar_filename = join(hpss_path, f"{year}", f"day{jday:03}", f"OR_{instrument}_g16_{sector}_{date_str}_{hour:02}{ending}")
                subprocess.call(["hsi", "get", tar_filename])
    except Exception as e:
        print(traceback.format_exc())
        raise e
    return 0


def extract_tar_file(tar_file, out_path):
    return
if __name__ == "__main__":
    main()

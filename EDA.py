# Import libraries
import sys
sys.path.append('C:\\Users\\spong\\Documents\\DSAI\\ConvScripts')
                
import ForBeginning as fb
import modules as md
import os
import pandas as pd
import polars as pl
import dtale as dt
import zipfile as zf

from tqdm import tqdm
import time
from datetime import datetime

# Functions and classes


# Main program
if __name__ == '__main__':

    print('='*20 + f' {datetime.now().replace(microsecond=0)} Program Start ' + '='*20 +'\n')
    start = time.time()
    dir_mydoc = fb.ChangeDir()

    usrIn = False
    df = md.CheckEEG('./train.csv', './train_eegs.zip')
    df.to_csv('train_matchedEEG.csv', index=False)

    if usrIn == True:
        # Overview of train.csv
        df = fb.DescriptiveStat(file, file.split('.')[-1])

    end = time.time()

    print('='*20 + ' Program End ' + '='*20 + '\n')
    print(f'Execution time: {(end - start):.2f}s')

pass
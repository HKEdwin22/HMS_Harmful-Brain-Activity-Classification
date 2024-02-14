# Import libraries
import sys, os
sys.path.append('C:\\Users\\spong\\Documents\\DSAI\\ConvScripts')
                
import ForBeginning as fb
import modules as md
import pandas as pd
import polars as pl
import dtale as dt
import zipfile as zf

from tqdm import tqdm
import time
from datetime import datetime

# Functions and classes
class Config():
    usrIn = False
    rawPath = './rawData/'
    augPath = './augData/'

# Main program
if __name__ == '__main__':

    print('='*20 + f' {datetime.now().replace(microsecond=0)} Program Start ' + '='*20 +'\n')
    start = time.time()
    dir_mydoc = fb.ChangeDir()
    
     
    
    if Config.usrIn == True:

        # Overview of train.csv
        file = Config.rawPath + 'train.csv'
        df = fb.DescriptiveStat(file, file.split('.')[-1])

        # Check if EEG records match with EEG parquet
        eda = md.EDA()
        df = eda.SampleRecMatch('./train.csv', './train_eegs.zip')
        df.to_csv('train_matchedEEG.csv', index=False)

        # Count EEG sample number for each EEG id & EEG record number for each subject
        eda = md.EDA()
        eda.SamplingSize(Config.rawPath + 'train.csv')

        # Descriptive statistics for number of EEG samples for each EEG entry
        file = Config.augPath + 'EEG_sampleNumber.csv'
        df = fb.DescriptiveStat(file, file.split('.')[-1], True)

        # Descriptive statistics for number of EEG entry for each subject
        file = Config.augPath + 'patient_eegNumber.csv'
        df = fb.DescriptiveStat(file, file.split('.')[-1], True)

    end = time.time()

    print('='*20 + ' Program End ' + '='*20 + '\n')
    print(f'Execution time: {(end - start):.2f}s')

pass
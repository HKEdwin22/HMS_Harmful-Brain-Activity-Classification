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
    
    # Descriptive statistics for number of EEG samples for each EEG entry
    file = Config.augPath + 'EEG_sampleNumber.csv'
    df = fb.DescriptiveStat(file, file.split('.')[-1])

    # Visualise analysis
    df = df[(df['sample_num'] <= 10) & (df['sample_num'] >= 2)]
    df = df.groupby(['sample_num', 'patient_id'])['eeg_id'].count().to_frame(name='eeg_entry_count')
    df.reset_index(inplace=True)

    for snum in df.sample_num:
        dfTmp = df[df.sample_num == snum]
        print('\n' + '='*20 + f'For sample number equals to {snum}')
        print('\n' + '='*20 + ' Number of unique values ' + '='*20)
        print(dfTmp.nunique())
        print('\n' + '='*20 + ' Descriptive statistics for each column ' + '='*20 + '\n')
        print(dfTmp.describe())
    
    if Config.usrIn == True:

        # Overview of train.csv
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
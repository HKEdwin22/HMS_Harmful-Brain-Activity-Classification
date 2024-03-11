# Import libraries
import sys
sys.path.append('C:\\Users\\spong\\Documents\\DSAI\\ConvScripts')
                
import ForBeginning as fb

import modules as md
import pandas as pd
import polars as pl

from tqdm import tqdm
import time
from datetime import datetime

import seaborn as sns
import matplotlib.pyplot as plt

# Functions and classes
class Config():
    seed = 73
    usrIn = False
    rawPath = './rawData/'
    augPath = './augData/'
    ExtEEGs = augPath + 'Extracted_EEGs/'

# Main program
if __name__ == '__main__':

    print('='*20 + f' {datetime.now().replace(microsecond=0)} Program Start ' + '='*20 +'\n')
    start = time.time()
    dir_mydoc = fb.ChangeDir()
    spp = md.SignalPreprocessing()

    # Extract the central 10-second of each subsample
    zPath = Config.rawPath + 'train_eegs2.zip'
    file1 = Config.augPath + 'thousand_subsamples_per_type.csv'
    nPath = Config.ExtEEGs
    spp.Extract10s(zPath, file1, nPath)

    '''
    Denoising EEG subsamples
    '''
    # import pywt
    # from math import sqrt, log10

    # file = Config.augPath + './thousand_subsamples_per_type.csv'
    # df = pd.read_csv(file)

    # for rows in tqdm(df.index):
    #     eid = df.iloc[rows, 0]
    #     subsample = df.iloc[rows, 1]
    #     pFile = Config.ExtEEGs + f'{eid}_{subsample}.parquet'
    #     x = pl.read_parquet(pFile).to_pandas()
    #     x = x.iloc[:, :-1]
        
    #     # Step 1 - estimate the approximated and multilelvel detailed coefficients
    #     db4 = pywt.Wavelet('db4')
    #     cA4, cD5, cD4, cD3, cD2, cD1 = pywt.wavedec(x, db4, mode='periodic', level=5)

    #     # Step 2 - compute sigma value & estimate the threshold
    #     sigma = fb.MAD(cD5)/.6745
    #     threshold = sigma * sqrt(2*log10(len(x)))
    #     fCoeff = pywt.threshold(x, threshold, 'soft')

    if Config.usrIn == True:
        
        '''
        PART 3 - CREATE DATASET FOR DENOISING
        '''


        '''
        PART 3 - CREATE DATASET FOR DENOISING (ABANDONED)
        '''
        # Create dataset for denoising
        file = Config.augPath + 'EEG_sampleNumber.csv'
        newf = Config.augPath + 'eid_for_training.csv'
        spp.TrainingEID(file, Config.seed, newf)

        # Check if the new dataset has duplicated information
        file = Config.augPath + 'eid_for_training.csv'
        df = pd.read_csv(file)
        print(f'There are {len(df)} entries. Unique entries as below:\n')
        print(df.nunique())

    end = time.time()
    print('='*20 + f' Program End {datetime.now().replace(microsecond=0)}' + '='*20)
    print(f'Execution time: {(end - start):.2f}s')

pass
# Import libraries
import sys, os
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

# Main program
if __name__ == '__main__':

    print('='*20 + f' {datetime.now().replace(microsecond=0)} Program Start ' + '='*20 +'\n')
    start = time.time()
    dir_mydoc = fb.ChangeDir()
    spp = md.SignalPreprocessing()

    if Config.usrIn == True:
        
        '''
        PART 3 - CREATE DATASET FOR DENOISING
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
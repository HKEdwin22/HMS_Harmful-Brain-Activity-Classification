# Import libraries
import sys
sys.path.append('C:\\Users\\spong\\Documents\\DSAI\\ConvScripts')
                
import ForBeginning as fb

import modules as md
from modules import Config
import pandas as pd
import polars as pl
import numpy as np

from tqdm import tqdm
import time
import pickle
from datetime import datetime

import seaborn as sns
import matplotlib.pyplot as plt

# Main program
if __name__ == '__main__':

    print('='*20 + f' {datetime.now().replace(microsecond=0)} Program Start ' + '='*20 +'\n')
    start = time.time()
    dir_mydoc = fb.ChangeDir()
    spp = md.SignalPreprocessing()
    denoise = md.Denoising()
    specgram = md.Spectrogram()

    '''Dataset preparation'''
    
    # Read the training set
    with open(Config.augPath + "spectrogram_all.pkl", "rb") as f:
        dfTgt = pickle.load(f)

    dfTgt = pd.DataFrame(dfTgt)

    if Config.usrIn == True:
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
    print('='*20 + f' Program End {datetime.now().replace(microsecond=0)} ' + '='*20)
    print(f'Execution time: {(end - start):.2f}s')

pass
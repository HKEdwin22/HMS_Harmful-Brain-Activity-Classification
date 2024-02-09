# Import libraries
import sys
sys.path.append('C:\\Users\\spong\\Documents\\DSAI\\ConvScripts')

import ForBeginning as fb
import pandas as pd
import polars as pl
from tqdm import tqdm

def CheckEEG(f, zPath):
    '''
    Check if all EEG records exist in train.csv and return a new .csv file that holds all EEG samples
    f : the file outside the zip archive
    zPath : the path of the zip archive
    '''

    df = pd.read_csv(f)
    validEid = []
    
    for eid in tqdm(df['eeg_id'].unique()):

        # Construct the file name of the target file in the zip archive
        pFile = str(eid) + '.parquet'
        
        # Read the file and return a polars dataframe
        pldf = fb.NoNeedUnzip(pFile, zPath, 'parquet')

        recTime = int(df[df['eeg_id'] == eid]['eeg_label_offset_seconds'].iloc[-1]) + 50
        sampleTime = pldf.shape[0] / 200

        if recTime == sampleTime:
            validEid.append(eid)
    
    return df[df['eeg_id'].isin(validEid)]
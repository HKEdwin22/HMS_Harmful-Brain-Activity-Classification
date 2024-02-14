# Import libraries
import sys
sys.path.append('C:\\Users\\spong\\Documents\\DSAI\\ConvScripts')

import ForBeginning as fb
import pandas as pd
import polars as pl
from tqdm import tqdm

class EDA():
    '''
    This class carries out Exploratory Data Analysis for .parquet files
    fileName : the file outside the zip archive
    zipPath : the path of the zip archive
    '''
        
    def __init__(self, fileName, zipPath) -> None:
        self.f = fileName
        self.zPath = zipPath

        pass

    def SampleRecMatch(self):
        '''
        Check if all EEG records exist in train.csv and return a new .csv file that holds all EEG samples
        '''

        df = pd.read_csv(self.f)
        validEid = []
        
        for eid in tqdm(df['eeg_id'].unique()):

            # Construct the file name of the target file in the zip archive
            pFile = str(eid) + '.parquet'
            
            # Read the file and return a polars dataframe
            pldf = fb.NoUnzip(pFile, self.zPath, 'parquet')

            # Compute the recorded duration and sampled duration
            recTime = int(df[df['eeg_id'] == eid]['eeg_label_offset_seconds'].iloc[-1]) + 50
            sampleTime = pldf.shape[0] / 200

            if recTime == sampleTime:
                validEid.append(eid)
        
        return df[df['eeg_id'].isin(validEid)]
    
    def SbjSampleNum(self, file):
        '''
        Check the number of samples extracted for each subject
        file: the name of the csv file
        '''
        df = pd.read_csv(file)
        eeg, snum = [], []
        pid, enum = [], []

        for eid in tqdm(df['eeg_id'].unique()):
            eeg.append(eid)
            snum.append(df[df.eeg_id == eid].count())

        dfNew = pd.DataFrame({
            'eeg_id': eeg,
            'sample num': snum
            })
        
        dfNew.to_csv('./augData/EEG_sampleNumber.csv')

        for patient in tqdm(df['patient_id'].unique()):
            pid.append(patient)
            enum.append(df[df.patient_id == patient].count())

        dfNew = pd.DataFrame({
            'patient_id' : pid,
            'eeg_num' : enum
        })

        dfNew.to_csv('./augData/patient_eegNumber.csv')
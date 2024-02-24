# Import libraries
import sys
sys.path.append('C:\\Users\\spong\\Documents\\DSAI\\ConvScripts')

import ForBeginning as fb
import random
import pandas as pd
import polars as pl
import pywt
from tqdm import tqdm

class EDA():
    '''
    This class carries out Exploratory Data Analysis for .parquet files
    '''
        
    def __init__(self) -> None:
        pass

    def SampleRecMatch(self, f, zPath):
        '''
        Check if all EEG records exist in train.csv and return a new .csv file that holds all EEG samples
        fileName : the file outside the zip archive
        zipPath : the path of the zip archive
        '''

        df = pd.read_csv(f)
        validEid = []
        
        for eid in tqdm(df['eeg_id'].unique()):

            # Construct the file name of the target file in the zip archive
            pFile = str(eid) + '.parquet'
            
            # Read the file and return a polars dataframe
            pldf = fb.NoUnzip(pFile, zPath, 'parquet')

            # Compute the recorded duration and sampled duration
            recTime = int(df[df['eeg_id'] == eid]['eeg_label_offset_seconds'].iloc[-1]) + 50
            sampleTime = pldf.shape[0] / 200

            if recTime == sampleTime:
                validEid.append(eid)
        
        return df[df['eeg_id'].isin(validEid)]
    
    def SamplingSize(self, file):
        '''
        Check the number of samples extracted for each subject
        file: the name of the csv file
        '''
        df = pd.read_csv(file)
        eeg, pat, snum = [], [], []
        pid, enum = [], []

        # Check the sampling size for each EEG entry
        for eid in tqdm(df['eeg_id'].unique()):
            tmp = max(df[df.eeg_id == eid]['eeg_sub_id']) + 1
            pat.append(df[df.eeg_id == eid]['patient_id'].unique()[0])
            eeg.append(eid)
            snum.append(tmp)

        dfNew = pd.DataFrame({
            'eeg_id': eeg,
            'patient_id': pat,
            'sample num': snum
        })
        
        dfNew.to_csv('./augData/EEG_sampleNumber.csv', index=False)

        # Use the new dataframe to count the number of EEG entries for each patient
        for patient in tqdm(dfNew['patient_id'].unique()):
            tmp = len(dfNew[dfNew.patient_id == patient])
            pid.append(patient)
            enum.append(tmp)

        dfNew2 = pd.DataFrame({
            'patient_id' : pid,
            'eeg_num' : enum
        })

        dfNew2.to_csv('./augData/patient_eegNumber.csv', index=False)

class SignalPreprocessing():
    
    def __init__(self) -> None:
        pass

    def TrainingEID(self, file, seed):
        '''
        Extract eeg_id of the EEG signals that contain at least 10 subsamples and form a sample pool (A) &
        draw one eeg_id from sample pool (A)
        file: Config.augPath + 'EEG_sampleNumber.csv'
        seed: Config.seed
        '''

        df = pd.read_csv(file)
        df = df[df.sample_num >= 10]
        random.seed(seed)
        resultEid = []
        
        # Check if there are duplicated eeg_id
        if df['eeg_id'].nunique() == len(df):
            print('Confirmed all eeg_id are unique.')
        else:
            print('There are eeg_id duplicated.')

        # Randomly draw one signal from each subject    
        for pid in tqdm(df['patient_id'].unique()):
            eid = df[df.patient_id == pid].eeg_id
            eid = eid.to_list()
            if len(eid) == 1:
                resultEid.append(eid[0])
            else:
                rint = random.randint(0, len(eid) - 1)
                resultEid.append(eid[rint])

        df = df.set_index('eeg_id')
        df = df.loc[resultEid]
        df = df.reset_index()
        df.to_csv('./augData/eid_for_training.csv', index=False)   

    def Denoising(self):
        '''
        This function serves as algorithm 1 in reference [1]
        '''
        pass

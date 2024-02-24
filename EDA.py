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

    file = Config.augPath + 'rawData_with_case.csv'
    case = 'ideal'

    '''
    Extract ideal cases
    '''
    df = pd.read_csv(file)
    df = df[df.case == 'ideal']
    df = df.groupby(['patient_id', 'eeg_id']).size()
    df = df.to_frame().reset_index()
    df.columns = ['patient_id', 'eeg_id', 'sample_num']





    
            
    

    if Config.usrIn == True:

        '''
        PART 1 - STUDY THE EEG DATA
        '''

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

        # Descriptive statistics for the distribution of labels and cases
        file = Config.rawPath + 'train.csv'
        md.EDA.Case(file)
        file = Config.augPath + 'rawData_with_case.csv'
        md.EDA.LabelDistribution(file)

        '''
        PART 2 - CREATE DATASET FOR DENOISING
        '''
        # Create dataset for denoising
        Spp = md.SignalPreprocessing()
        file = Config.augPath + 'EEG_sampleNumber.csv'
        Spp.TrainingEID(file, Config.seed)

        # Check if the new dataset has duplicated information
        file = Config.augPath + 'eid_for_training.csv'
        df = pd.read_csv(file)
        print(f'There are {len(df)} entries. Unique entries as below:\n')
        print(df.nunique())

    end = time.time()

    print('='*20 + f' Program End {datetime.now().replace(microsecond=0)}' + '='*20)
    print(f'Execution time: {(end - start):.2f}s')

pass
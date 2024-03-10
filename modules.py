# Import libraries
import sys
sys.path.append('C:\\Users\\spong\\Documents\\DSAI\\ConvScripts')

import ForBeginning as fb
import random
import pandas as pd
import polars as pl

import pywt

import matplotlib.pyplot as plt
import seaborn as sns
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

    def Case(rawfile):
        '''
        Add column 'case' to the raw data
        rawfile : './rawData/train.csv'
        '''
        
        df = pd.read_csv(rawfile)
        case = []
        ideal, proto, edge = 0, 0, 0

        for row in tqdm(df.index):
            vote = df.iloc[row, 9:]
            totalVote = vote.sum()
            otherVote = vote.other_vote
            ratio = round(otherVote / totalVote, 1)

            if ratio == 0.0:
                case.append('ideal')
                ideal += 1
            elif ratio >= 0.5:
                case.append('proto')
                proto += 1
            else:
                case.append('edge')
                edge += 1

        df['case'] = case
        data = {'case': ['ideal', 'proto', 'edge'],
                'quantity': [ideal, proto, edge]
                }
        data = pd.DataFrame(data)
        print(data)

        df.to_csv('./augData/rawData_with_cases.csv', index=False)

    def LabelDistribution(augfile):

        '''
        EDA: rawData_with_cases.csv
        augfile : './augPath/rawData_with_cases.csv'
        '''
        
        df = pd.read_csv(augfile)
        label = df.expert_consensus.unique()
        label = label.tolist()

        df = df.groupby(['expert_consensus', 'case']).size()
        df = df.to_frame()
        df = df.reset_index()
        df.columns = ['expert_consensus', 'case', 'quantity']

        gf = sns.barplot(x='expert_consensus', y='quantity', hue='case', data=df)
        gf.axhline(7971, color='black', linestyle='--')
        plt.text(5.6, 7971, '7971', verticalalignment='center')
        plt.title('Distribution of Class Label by Case')
        plt.xlabel('Class Label')
        plt.show()

class SignalPreprocessing():
    
    def __init__(self) -> None:
        pass

    def TrainingSet(self, file, seed, newfile):
        '''
        Construct training set with 1000 subsamples per type
        file: Config.augPath + 'rawData (ideal & proto)_without_DiscontinuedEEG.csv'
        seed: Config.seed
        newfile : name of the new file (Config.augPath + 'thousand_subsamples_per_type.csv')
        '''       
        
        dfRaw = pd.read_csv(file)
        random.seed(seed)
        df = pd.DataFrame().reindex_like(dfRaw)    

        # Create a dataframe to hold the 1000 subsamples
        for sampleType in dfRaw['expert_consensus'].unique():
            dfSpecificType = dfRaw[dfRaw.expert_consensus == sampleType]
            dfSpecificType = dfSpecificType.reset_index(drop=True)
            randomIdx = [random.choice(dfSpecificType.index) for _ in range(1000)]
            df = pd.concat([df, dfSpecificType.iloc[randomIdx, :]], ignore_index=True, sort=False)

        df.dropna(inplace=True)
        df.to_csv(newfile) 

    def LabelBalance(self, file, case):
        '''
        Purpose: extract subsamples that have changed brain activities and return clean raw data with ideal cases only
        file: ./augData/rawData_with_cases.csv
        case: 'ideal'/'edge'/'proto'
        '''

        # Check class balance
        dfRaw = pd.read_csv(file)
        dfRaw = dfRaw[dfRaw.case == case]
        diff = []

        for eid in tqdm(dfRaw['eeg_id'].unique()):
            t = dfRaw[dfRaw.eeg_id == eid]['expert_consensus']
            t = t.to_list()

            # Check if the target labels are consistant
            if case == 'ideal' and len(t) != t.count(t[0]):
                diff.append(eid)
            elif case == 'proto':
                if 'Other' in t == False:
                    diff.append(eid)
                elif t.count('Other') != len(t):
                    diff.append(eid)

        print(f'There are {len(diff)} EEG signals having inconsistent labels.')

        dfDis = pd.DataFrame({'eeg_id': diff})
        dfDis.to_csv('./augData/DiscontinueEEG.csv')

        # Remove eeg_id that has discontinued brain activities  
        dfRaw.set_index('eeg_id', inplace=True)
        dfRaw.drop(dfDis.eeg_id.to_list(), inplace=True)
        dfRaw = dfRaw.reset_index()
        dfRaw.to_csv('./augData/RawData_without_DiscontinuedEEG.csv', index=False)
        
    def Extract10s(self, zPath, file1, nPath):
        '''
        Extract the ten-second that the experts annotated and save the extracted signal to a new file
        zPath: the path of the zip archive
        file1: thousand_subsamples_per_type.csv
        nPath: new path for the new files
        '''
        
        df = pd.read_csv(file1)

        for idx in tqdm(df.index):
            eid = df.iloc[idx, 0]
            subsample = df.iloc[idx, 1]
            t0 = df.iloc[idx, 2]
            s0 = 200*t0
            z0 = int(200*50/2 + s0*200 - 1000)
            z1 = int(200*50/2 + s0*200 + 1000)

            pFile = str(eid) + '.parquet'
            newFile = nPath + f'{eid}_{subsample}.parquet'
            dfSignal = fb.NoUnzip(pFile, zPath, 'parquet')
        
            # Select the 10-second signals in the middle of the subsample
            selectedRow = dfSignal.slice(z0, z1)
            selectedRow.write_parquet(newFile)

    def Denoising(self):
        '''
        This function serves as algorithm 1 in reference [1]
        '''
        pass

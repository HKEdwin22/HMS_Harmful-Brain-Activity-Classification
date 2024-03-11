# Import libraries
import sys
sys.path.append('C:\\Users\\spong\\Documents\\DSAI\\ConvScripts')

import ForBeginning as fb
import random
import pandas as pd
import polars as pl
import numpy as np

import pywt
from math import sqrt, log10
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

# Configurations
class Config():
    seed = 73
    usrIn = False
    sampleSignals = './SampleSignals/'
    rawPath = './rawData/'
    augPath = './augData/'
    ExtEEGs = augPath + 'Extracted_EEGs/'
    DenoisedEEGs = augPath + 'Denoised_EEGs/'

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
            idxList = dfSpecificType.index
            idxList = random.sample(idxList.to_list(), 1000)
            df = pd.concat([df, dfSpecificType.iloc[idxList, :]], ignore_index=True, sort=False)

        df.dropna(inplace=True)
        df.to_csv(newfile, index=False) 

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
            z0 = int(200*50/2 + s0 - 1000)

            pFile = str(eid) + '.parquet'
            newFile = nPath + f'{eid}_{subsample}.parquet'
            dfSignal = fb.NoUnzip(pFile, zPath, 'parquet')
        
            # Select the 10-second signals in the middle of the subsample
            selectedRow = dfSignal.slice(z0, 2000)
            selectedRow.write_parquet(newFile)

class Denoising():
        '''
        This class serves as algorithm 1 in reference [1]
        '''
        def __init__(self) -> None:
            pass

        def Thresholding(self, wave, oriSignal, mode='soft'):
            '''
            Compute sigma & threshold value. Return filtered Coefficient which are greater than the threshold.
            wave: detail coefficient (e.g. cD5)
            oriSignal: input signal x
            mode: hard or soft (for thresholding)
            '''

            sigma = fb.MAD(wave)/.6745
            threshold = sigma * sqrt(2*log10(len(oriSignal)))
            fCoeff = pywt.threshold(wave, threshold, mode=mode)

            return fCoeff

        def DenoiseProcess(self, file):
            '''
            Algorithm 1 - Denoising EEG subsamples
            file: dataset (thousand_subsamples_per_type.csv)
            '''
            df = pd.read_csv(file)

            for rows in tqdm(df.index):
                eid = df.iloc[rows, 0]
                subsample = df.iloc[rows, 1]
                pFile = Config.ExtEEGs + f'{eid}_{subsample}.parquet'
                x = pl.read_parquet(pFile).to_pandas()
                x = x.iloc[:, :-1]
                
                # Step 1 - estimate the approximated and multilelvel detailed coefficients
                db4 = pywt.Wavelet('db4')
                L = pywt.wavedec(x, db4, mode='periodic', level=5)

                # Step 2, 3 & 4 - compute sigma value, estimate the threshold & restore the signal
                F = [L[0]]
                for i in range(1,len(L)):
                    F.append(self.Thresholding(L[i], x))

                Rsignal = pywt.waverec(F, db4, mode='periodic')
                
                # Step 5 - Save the restored signal as a numpy file
                newFile = Config.DenoisedEEGs + f'{eid}_{subsample}_denoised.npy'
                np.save(newFile, Rsignal)

        def VisualiseSignals(self, eidSample):
            '''
            Visualise the signals before and after denoising
            eidsample: sample ID (e.g. '525664301_444')
            '''

            fileRaw = Config.ExtEEGs + f'{eidSample}.parquet'
            fileDenoise = Config.DenoisedEEGs + f'{eidSample}_denoised.npy'
            signalRaw = pl.read_parquet(fileRaw).to_pandas()
            signalDenoise = np.load(fileDenoise)

            features = signalRaw.columns

            for f in range(len(features)):
                file = Config.sampleSignals + f'{eidSample}_{features[f]}.jpg'
                plt.plot(signalRaw[features[f]], color='black', label='Raw Signal')
                plt.plot(signalDenoise[:,f], color='red', label='Denoised Signal')
                plt.title(features[f])
                plt.xlabel('Samples')
                plt.ylabel('Amplitude')
                plt.savefig(file)
                plt.clf()

            plt.plot(signalRaw, color='black')
            plt.plot(signalDenoise)
            plt.title(f'Raw and Denoised Signals for {eidSample}')
            plt.savefig(Config.sampleSignals + f'{eidSample}.jpg')
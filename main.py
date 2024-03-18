# Import libraries
import sys
sys.path.append('C:\\Users\\spong\\Documents\\DSAI\\ConvScripts')
                
import ForBeginning as fb

import modules as md
from modules import Config
import pandas as pd
import numpy as np
import polars as pl

from tqdm import tqdm
import time
from datetime import datetime

import seaborn as sns
import matplotlib.pyplot as plt

# Main program
if __name__ == '__main__':

    print('='*20 + f' {datetime.now().replace(microsecond=0)} Program Start ' + '='*20 +'\n')
    start = time.time()
    dir_mydoc = fb.ChangeDir()
    spp = md.SignalPreprocessing()

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
    file = Config.augPath + 'rawData_with_cases.csv'
    md.EDA.LabelDistribution(file)

    '''
    PART 2 - CREATE A BALANCE TRAINING SET
    '''
    # Identify & remove signals with changed brain activities & keep 'ideal' case
    file = Config.augPath + 'rawData_with_cases.csv'
    spp.LabelBalance(file, 'ideal')
    spp.LabelBalance(file, 'proto')

    # Merge the csv files into one
    f1 = Config.augPath + 'rawData (ideal case)_without_DiscontinuedEEG.csv'
    f2 = Config.augPath + 'rawData (proto case)_without_DiscontinuedEEG.csv'
    df1 = pd.read_csv(f1)
    df2 = pd.read_csv(f2)
    df = pd.concat([df1, df2], ignore_index=True, sort=False)
    df.to_csv('./augData/rawData (ideal & proto)_without_DiscontinuedEEG.csv', index=False)

    # Plot the distribution of all types within ideal (for the 5 types) & proto cases (for Other)
    gf = sns.countplot(df, x='expert_consensus', order=['GPD', 'GRDA', 'LPD', 'LRDA', 'Other', 'Seizure'])
    gf.axhline(6559, color='black', linestyle='--')
    plt.text(5.6, 6559, '6559', verticalalignment='center')
    plt.show()

    '''
    PART 3 - CREATE DATASET FOR DENOISING
    '''
    # Randomly draw 1000 signals for each class
    file = Config.augPath + 'rawData (ideal & proto)_without_DiscontinuedEEG.csv'
    newFile = Config.augPath + 'thousand_subsamples_per_type.csv'
    spp.TrainingSet(file, Config.seed, newFile)

    zPath = Config.rawPath + 'train_eegs.zip'
    file1 = Config.augPath + 'thousand_subsamples_per_type.csv'
    nPath = Config.ExtEEGs
    spp.Extract10s(zPath, file1, nPath)

    # Algorithm 1 - Denoising EEG subsamples
    denoise = md.Denoising()
    file = Config.augPath + './thousand_subsamples_per_type.csv' # input 1
    denoise.DenoiseProcess(file)

    # Algorithm 2 - Frequency filtration
    w, f = 130, 80
    file = Config.augPath + 'thousand_subsamples_per_type.csv'
    denoise.FrequencyFiltration(file, w, f)

    # Visualise the denoised/filtered signals
    eidsample = '1248563466_1'
    visualise = md.VisualiseSignal()
    visualise.TimeDomainGraph(eidsample, 'filtrated')
    visualise.FreqDomainGraph(eidsample, 'filtrated')

    # Generate spectrogram
    windowLength = 130
    freq = 80
    eidsample = '1248563466_1'
    file = Config.readyset + f'w130f80/FilteredFreq/{eidsample}_filtrated.npy'
    signal = np.load(file)
    rSgn = []

    features = pl.read_parquet(Config.rawPath + '2061593eeg.parquet').columns[:-1]

    for col in range(signal.shape[1]):
        x = signal[:1000, col]
        f, t, Z = visualise.Spectrogram(x, features[col], sf=freq, n=windowLength)
        rSgn.append(Z)
        plt.clf()

    meanSgn = np.mean(np.abs(rSgn), axis=0)
    graph = plt.pcolormesh(t, np.abs(f), np.abs(meanSgn), shading='gouraud', vmin=0, vmax=1)
    plt.colorbar(graph)

    plt.title(f'Averaged Spectrogram for {eidsample}')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time')

    plt.savefig(Config.augPath + f'Averaged_Spectrogram_{eidsample}.jpg')
    plt.show()

    end = time.time()
    print('='*20 + f' Program End {datetime.now().replace(microsecond=0)}' + '='*20)
    print(f'Execution time: {(end - start):.2f}s')
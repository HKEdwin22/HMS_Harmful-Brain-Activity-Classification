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

    # Algorithm 2 - Frequency filtration
    file = Config.augPath + 'thousand_subsamples_per_type.csv'
    denoise.FrequencyFiltration(file, 400)
    
    '''
    Plot the Power Spectrum
    '''
    from scipy.fft import fft, fftfreq, ifft

    eidsample = '1480985066_196'

    denoise.VisualiseSignals(eidsample, 'filtrated')
    
    file = Config.ExtEEGs + f'{eidsample}.parquet'
    signalRaw = pl.read_parquet(file).to_pandas()
    signalRaw = signalRaw.iloc[:,:-1]
    
    fileFiltered = Config.PowerSpectrum + f'{eidsample}_PowerSpectrum.npy'
    SProcessed = np.load(fileFiltered)

    features = signalRaw.columns

    for f in range(len(features)):
        rawPower = 10*np.log10(np.abs(signalRaw[features[f]])**2 + 1e-9)
        proPower = 10*np.log10(np.abs(SProcessed[:, f])**2 + 1e-9)

        Sfreq = fftfreq(2000, d=1/200)
        idx = np.where((Sfreq<=30) & (Sfreq>=0))[0]
        
        plt.clf()
        plt.plot(Sfreq[idx], rawPower[idx], color='black', label='Raw Signal')
        plt.plot(Sfreq[idx], proPower[idx], color='red', label='Processed Signal')
        plt.legend()
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power (μV²)")
        plt.title(f"EEG Power Spectrum of brain region {features[f]} for {eidsample}")

        figManager = plt.get_current_fig_manager()
        figManager.window.state('zoomed')
        plt.savefig(Config.augPath + f'{eidsample}_{features[f]}.jpg')


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
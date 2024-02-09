# Import libraries
import sys
sys.path.append('C:\\Users\\spong\\Documents\\DSAI\\ConvScripts')

import ForBeginning as fb
import pandas as pd
import polars as pl
fb.ChangeDir()

df = pd.read_csv('./augData/train_matchedEEG.csv')

pass
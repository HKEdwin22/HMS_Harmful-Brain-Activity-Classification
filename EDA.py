# Import libraries
import sys
sys.path.append('C:\\Users\\spong\\Documents\\DSAI\\ConvScripts')
                
import ForBeginning as fb
import os
import pandas as pd
import dtale as dt

# Functions and classes


# Main program
if __name__ == '__main__':
    
    dir_mydoc = fb.ChangeDir()

    # Overview of train.csv
    file = './train.csv'
    df = fb.DescriptiveStat(file, 'csv')

pass
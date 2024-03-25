# Import libraries
from modules import Config
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn


'''
Dataset preparation
'''
# Read the training set
with open(Config.augPath + "spectrogram_all.pkl", "rb") as f:
    dfTgt = pickle.load(f)

# Split the data into training and test set
dfTgt = pd.DataFrame(dfTgt)
X = dfTgt.iloc[:, 3]
y = dfTgt.iloc[:, 4]
del dfTgt

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=Config.seed)

'''
Model Implementation
'''
class HMSConvNN(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(HMSConvNN, self).__init__(*args, **kwargs)
        self.Conv1 = nn.Conv2d(1, 256, kernel_size=(5,3), stride=(1,1), padding=0)
        self.Conv21 = nn.Conv2d(256, 512, kernel_size=(3,3), stride=(2,1), padding=0)
        self.Conv2 = nn.Conv2d(256, 512, kernel_size=(3,3), stride=(1,1), padding=0)
        self.Conv31 = nn.Conv2d(512, 1024, kernel_size=(3,3), stride=(2,1), padding=0)
        self.Conv3 = nn.Conv2d(512, 1024, kernel_size=(3,3), stride=(1,1), padding=0)
        self.maxPool = nn.MaxPool2d(kernel_size=(2,2), stride=1)
        self.relu = nn.ReLU(inplace=True)

        self.fc = nn.Linear(in_features=4*21*1024, out_features=6)

    def forward(self, x):
        
        # Hidden layer 1
        x = self.relu(self.Conv1(x))
        x = self.relu(self.Conv1(x))
        x = self.relu(self.Conv1(x))
        x = self.maxPool(x)

        # Hidden layer 2
        x = self.relu(self.Conv21(x))
        x = self.relu(self.Conv2(x))
        x = self.relu(self.Conv2(x))
        x = self.maxPool(x)

        # Hidden layer 3
        x = self.relu(self.Conv31(x))
        x = self.relu(self.Conv3(x))
        x = self.relu(self.Conv3(x))
        x = self.maxPool(x)

        # Fully connected layer
        x = torch.flatten(x)
        x = self.fc(x)
        out = {'out': x}

        return out
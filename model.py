# Import libraries
import sys
sys.path.append('C:\\Users\\spong\\Documents\\DSAI\\ConvScripts')

import ForBeginning as fb
from modules import Config

import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils

from tqdm import tqdm
import time
from datetime import datetime

def DatasetPreparation():
    '''
    Dataset preparation
    '''
    # Read the training set
    with open(Config.augPath + "spectrogram_all.pkl", "rb") as f:
        dfTgt = pickle.load(f)
    dfTgt = pd.DataFrame(dfTgt)

    # Encode the labels
    code = dfTgt['Class'].astype('category')
    coding = dict(enumerate(code.cat.categories))
    dfTgt['Class'] = code.cat.codes

    # Transform the data to tensors
    X = dfTgt.iloc[:, 3]
    X = torch.tensor(X).type(torch.float32)
    y = torch.tensor(dfTgt.iloc[:, 4].values).type(torch.long)

    X = X.unsqueeze(1)
    tensorDataset = data_utils.TensorDataset(X, y)

    trainSet, testSet = data_utils.random_split(tensorDataset, [int(len(X)*.8), int(len(X)*.2)])
    trainLoader = data_utils.DataLoader(trainSet, batch_size=Config.batchSize, shuffle=True)
    testLoader = data_utils.DataLoader(testSet, batch_size=Config.batchSize, shuffle=False)

    return trainLoader, testLoader, coding

'''
Model Implementation
'''
class HMSConvNN(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(HMSConvNN, self).__init__(*args, **kwargs)
        self.Conv11 = nn.Conv2d(1, 128, kernel_size=(5,3), stride=(1,1), padding=0)
        self.Conv1 = nn.Conv2d(128, 128, kernel_size=(5,3), stride=(1,1), padding=0)
        self.Conv21 = nn.Conv2d(128, 256, kernel_size=(3,3), stride=(2,1), padding=0)
        self.Conv2 = nn.Conv2d(256, 256, kernel_size=(3,3), stride=(1,1), padding=0)
        self.Conv31 = nn.Conv2d(256, 512, kernel_size=(3,3), stride=(2,1), padding=0)
        self.Conv3 = nn.Conv2d(512, 512, kernel_size=(3,3), stride=(1,1), padding=0)
        self.maxPool = nn.MaxPool2d(kernel_size=(2,2), stride=1)
        self.relu = nn.ReLU(inplace=True)

        self.fc = nn.Linear(in_features=4*21*512, out_features=6)

    def forward(self, x):
        
        # Hidden layer 1
        x = self.relu(self.Conv11(x))
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
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
    

print('='*20 + f' {datetime.now().replace(microsecond=0)} Program Start ' + '='*20 +'\n')
start = time.time()
dir_mydoc = fb.ChangeDir()

trainLoader, testLoader, coding = DatasetPreparation()

# Illustrate the input spectrograms
# import matplotlib.pyplot as plt
# train_features, train_labels = next(iter(trainLoader))
# print(f"Feature batch shape: {train_features.size()}")
# print(f"Labels batch shape: {train_labels.size()}")
# img = train_features[0].squeeze()
# label = train_labels[0]
# plt.imshow(img, cmap="gray")
# plt.show()
# print(f"Label: {label}")


'''
Model Training
'''
if torch.cuda.is_available(): 
 dev = "cuda:0" 
else: 
 dev = "cpu" 
device = torch.device(dev) 

cnn = HMSConvNN().to(device)
lossFn = nn.CrossEntropyLoss()
optimiser = optim.SGD(cnn.parameters(), lr=0.001, momentum=0.9)

for e in range(Config.nEpoch):

    for inputs, labels in trainLoader:
        inputs = inputs.type(torch.float32).to(device)
        labels = labels.to(device)
        predictions = cnn(inputs)
        loss = lossFn(predictions, labels)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    acc = 0
    count = 0
    
    for inputs, labels in testLoader:
        inputs = inputs.type(torch.float32).to(device)
        labels = labels.to(device)
        acc += (torch.argmax(predictions, 1) == labels).float().sum()
        count += len(labels)

    acc /= count    
    print(f'Epoch {e/Config.nEpoch}:\taccuracy: {acc:.4f}')

torch.save(cnn.state_dict(), 'model1.pth')

end = time.time()
print('='*20 + f' Program End {datetime.now().replace(microsecond=0)} ' + '='*20)
print(f'Execution time: {(end - start):.2f}s')

pass
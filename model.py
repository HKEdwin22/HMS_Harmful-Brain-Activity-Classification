# Import libraries
import sys
sys.path.append('C:\\Users\\spong\\Documents\\DSAI\\ConvScripts')

import ForBeginning as fb
from modules import Config

import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
from torchsummary import summary

from tqdm import tqdm
import time
from datetime import datetime

nEpoch = 150
batchSize = 100 #600 is too large.
learningRate = 1e-4
L2Lambda = 1e-10

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

    # Normalise the input spectrograms
    X = dfTgt.iloc[:, 3]
    scaler = MinMaxScaler()
    result = []
    for i in X:
       i = scaler.fit_transform(i)
       result.append(i)
    
    # Convert the data to tensors
    X = torch.tensor(result).type(torch.float32)
    y = torch.tensor(dfTgt.iloc[:, 4].values).type(torch.long)

    X = X.unsqueeze(1)
    tensorDataset = data_utils.TensorDataset(X, y)

    trainSet, testSet = data_utils.random_split(tensorDataset, [int(len(X)*.8), int(len(X)*.2)])
    trainLoader = data_utils.DataLoader(trainSet, batch_size=batchSize, shuffle=True)
    testLoader = data_utils.DataLoader(testSet, batch_size=batchSize, shuffle=False)

    return trainLoader, testLoader, coding

def VisualiseInputSpectrogram(x):
    '''
    Illustrate the input spectrograms
    x : train loader
    '''
    import matplotlib.pyplot as plt

    train_features, train_labels = next(iter(trainLoader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    img = train_features[0].squeeze()
    label = train_labels[0]
    plt.imshow(img, cmap="gray")
    plt.show()
    print(f"Label: {label}")


class HMSConvNN(nn.Module):
    '''
    Model Implementation
    '''
    def __init__(self, *args, **kwargs) -> None:
        super(HMSConvNN, self).__init__(*args, **kwargs)

        self.hiddenLayer1 = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=(5,3), stride=(1,1), padding=(0,1)),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,2), stride=1)
        )
        
        self.hiddenLayer2 = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=(3,3), stride=(1,1), padding=(0,1)),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,2), stride=1)
        )
        
        self.fc1 = nn.Linear(in_features=8*592*5, out_features=2048)
        self.fc2 = nn.Linear(in_features=2048, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=16)
        self.fc4 = nn.Linear(in_features=16, out_features=6)
        self.softmax = nn.Softmax(dim=1)

        self.weightInit()

    def weightInit(self):

        for _, seq in self._modules.items():
            for _, layer in seq._modules.items():
                if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias,0.0)
                    
        
        print('Weights initialised!')

    def forward(self, x):
        
        x = self.hiddenLayer1(x)
        x = self.hiddenLayer2(x)

        # Fully connected layers
        x = torch.flatten(x, 1)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = F.leaky_relu(self.fc4(x))

        return x
    

if __name__ == '__main__':
    
    print('='*20 + f' {datetime.now().replace(microsecond=0)} Program Start ' + '='*20 +'\n')
    start = time.time()
    dir_mydoc = fb.ChangeDir()

    torch.manual_seed(43)
    trainLoader, testLoader, coding = DatasetPreparation()

    '''
    Model Training
    '''
    if torch.cuda.is_available():
        dev = 'cuda:0'
        torch.cuda.manual_seed(43)
        torch.cuda.manual_seed_all(43)
    else:
        dev = "cpu"

    device = torch.device(dev) 

    cnn = HMSConvNN().to(device)
    summary(cnn, (1, 600, 7))

    lossFn = nn.CrossEntropyLoss()
    optimiser = optim.AdamW(cnn.parameters(), lr=learningRate, weight_decay=0)

    lossTable = pd.DataFrame(columns=['Training Loss', 'Training Accuracy', 'Validation Loss', 'Validation Accuracy'])
    for e in range(0, nEpoch, 1):

        trainingLoss = 0.0
        trainingAcc = 0.0
        validationAcc = 0.0
        validationLoss = 0.0
        total = 0

        cnn.train()
        pbar = tqdm(trainLoader)
        for nBatch, data in enumerate(pbar):

            pbar.set_description(f'Epoch {e+1}/{nEpoch}')
            
            inputs, labels = data
            inputs = inputs.type(torch.float32).to(device)
            labels = labels.to(device)
            
            # Model training
            l2_reg = 0
            predictions = cnn(inputs)
            loss = lossFn(predictions, labels)
            
            # L2 regularisation
            for param in cnn.parameters():
                l2_reg += torch.norm(param, p=2)**2
            loss += L2Lambda * l2_reg 

            # Backpropagation
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            # Accuracy & Loss
            trainingLoss += loss.item()
            total += labels.size(0)
            trainingAcc += (torch.argmax(cnn.softmax(predictions), 1) == labels).sum().item()

        trainingLoss /= len(trainLoader)
        trainingAcc /= total
        total = 0
        
        cnn.eval()
        with torch.no_grad():
            for inputs, labels in testLoader:

                inputs = inputs.type(torch.float32).to(device)
                labels = labels.to(device)
                
                predictions = cnn(inputs)
                validationLoss += lossFn(predictions, labels).item()
                validationAcc += (torch.argmax(cnn.softmax(predictions), 1) == labels).sum().item()
                total += len(labels)

        validationLoss /= len(testLoader)
        validationAcc /= total
        print(f'Training loss / accuracy: {trainingLoss:.4f} / {trainingAcc:.4f}' + ' '*10 + f'Validation loss / accuracy: {validationLoss:.4f} / {validationAcc:.4f}')    

        lossTable.loc[len(lossTable)] = [round(trainingLoss, 4), round(trainingAcc, 4), round(validationLoss, 4), round(validationAcc, 4)]
        
        
    torch.save(cnn.state_dict(), 'model1.pth')
    lossTable.to_csv('./loss table.csv', index=False)

    end = time.time()
    print('='*20 + f' Program End {datetime.now().replace(microsecond=0)} ' + '='*20)
    print(f'Execution time: {int((end - start) // 60)} min {int((end - start) % 60)} s')

    pass
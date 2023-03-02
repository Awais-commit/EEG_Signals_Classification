#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 12:10:22 2023

@author: fh
"""
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import os 
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


df = pd.read_csv('emotions.csv')


dfx = df.drop('label',axis=1)
dfy = df['label']

le = LabelEncoder()
dfy = le.fit_transform(dfy)

X_train,X_test, y_train,y_test = train_test_split(dfx, dfy, test_size = 0.1)

clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))

running_loss



############Neural Network#######################

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'using {device} device')


########## Data Set#########
X_train = torch.tensor(X_train, dtype=torch.float)
y_train = torch.tensor(labels_, dtype=torch.float)


dataloader = TensorDataset(X_train, y_train)
dataloader = DataLoader(dataloader, batch_size=16)






class EEG(nn.Module):
    def __init__(self):
        super(EEG, self).__init__()
        self.flatten = nn.Flatten()
        
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2548,1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 3), 
            )
        def forward(self,x):
            x = self.flatten(x)
            logits = self.linear_relu_stack(x)
            return logits


model = EEG().to(device)
print(model)
        

################## Defining Loss Function##############3
import torch.optim as optim

criierion = nn.CrossEntropyLoss()
optmiizer = optim.SGD(model.parameters(), lr = 0.001, momentum=0.9)

######## Training loop ######
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in dataloader(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')



PATH = './EEG.pth'
torch.save(net.state_dict(), PATH)


import matplotlib.pyplot as plt
plt.plot(losses)




model.eval()
with torch.no_grad():
    predictions = model(torch.tensor(X_test, dtype=torch.float))




predictions = torch.argmax(predictions, -1)




print(classification_report(predictions, y_test))




_ = ConfusionMatrixDisplay.from_predictions(predictions, y_test, display_labels = le.classes_, 
                                            colorbar=False, cmap="plasma")


        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
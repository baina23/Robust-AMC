import numpy as np
import seaborn as sns
import _pickle as cPickle
import matplotlib.pyplot as plt
%matplotlib inline

import datetime
import warnings
import os, random
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from models import AUTOENCODER
from torch.utils.data import DataLoader, TensorDataset
from utilities import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Current Device: ", device)
print(torch.cuda.get_device_name(torch.cuda.current_device()))

filename = "RML2016.10b.dat"
snrs, mods, X, labels = process_data(filename)

model = AUTOENCODER()
model.to(device)
model_checkpoint = "AUTOENCODER.pt"

x_train, x_test, y_train, y_test, test_labels, test_idx = train_test_split(X, labels, mods, NN = curr_model)

train_dataset = TensorDataset(x_train, x_train)
test_dataset  = TensorDataset(x_test,  x_test)
batch_size = 256
TrainLoader = DataLoader(train_dataset, batch_size = batch_size, 
                         shuffle = False)
TestLoader  = DataLoader(test_dataset,  batch_size = batch_size, 
                         shuffle = False)
num_epochs = 100
criterion  = nn.MSELoss()
optimizer  = optim.SGD(model.parameters(), lr = 0.001)

best_val_acc = 0.

for epoch in range(num_epochs) :
    model.train()
    train_epoch_loss = 0.
    train_epoch_acc  = 0.
    for batch_idx, (data, labels) in enumerate(TrainLoader) :
        data   = data.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(data)
        loss    = criterion(outputs, labels)
        acc     = evaluate_accuracy(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_epoch_loss += loss.item()
        train_epoch_acc  += acc.item()
    
    with torch.no_grad() :
        model.eval()
        val_epoch_loss = 0.
        val_epoch_acc  = 0.
        
        for X_val_batch, y_val_batch in TestLoader :
            X_val_batch = X_val_batch.to(device)
            y_val_batch = y_val_batch.to(device)
            
            y_val_pred = model(X_val_batch)
            
            val_loss = criterion(y_val_pred, y_val_batch)
            val_acc  = evaluate_accuracy(y_val_pred, y_val_batch)
            
            val_epoch_loss += val_loss.item()
            val_epoch_acc  += val_acc.item()
    
    avg_train_loss = float(train_epoch_loss) / len(TrainLoader)
    avg_train_acc  = float(train_epoch_acc) / len(TrainLoader)
    avg_val_loss   = float(val_epoch_loss) / len(TestLoader)
    avg_val_acc    = float(val_epoch_acc) / len(TestLoader)
    
    print(f'Epoch {epoch+1}: | Train Acc: {avg_train_acc:.3f} | Test Acc: {avg_val_acc:.3f}')
    
    if avg_val_acc > best_val_acc :
        print("Saving Model Checkpoint......")
        best_val_acc = avg_val_acc
        torch.save(model.state_dict(), model_checkpoint)

print("Training Complete!")
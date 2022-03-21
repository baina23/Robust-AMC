import numpy as np
#import seaborn as sns
import _pickle as cPickle
#import matplotlib.pyplot as plt
#%matplotlib inline

import datetime
import warnings
import argparse 
import os, random
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from models import AUTOENCODER
from torch.utils.data import DataLoader, TensorDataset
from utilities import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", help="index of gpu", type=str, default=-1)
    return parser.parse_args()
args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Current Device: ", device)
print(torch.cuda.get_device_name(torch.cuda.current_device()))

filename = "RML2016.10b.dat"
snrs, mods, X, labels = process_data(filename)

model = AUTOENCODER()
model.to(device)
model_checkpoint = "AUTOENCODER_sub12.pt"

x_train, x_test, y_train, y_test, test_labels, test_idx = train_test_split(X, labels, mods)
x_train = x_train[:,:,:,::2]
x_test = x_test[:,:,:,::2]

train_dataset = TensorDataset(x_train, x_train)
test_dataset  = TensorDataset(x_test,  x_test)
batch_size = 256
TrainLoader = DataLoader(train_dataset, batch_size = batch_size, 
                         shuffle = False)
TestLoader  = DataLoader(test_dataset,  batch_size = batch_size, 
                         shuffle = False)
num_epochs = 100
criterion  = nn.MSELoss()
optimizer  = optim.SGD(model.parameters(), lr = 0.6, momentum = 0.9)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

best_val_loss = 1.

for epoch in range(num_epochs) :
    model.train()
    train_epoch_loss = 0.
    train_epoch_acc  = 0.
    for batch_idx, (data, lbs) in enumerate(TrainLoader) :
        data   = data.to(device)
        lbs = lbs.to(device)
        
        optimizer.zero_grad()
        outputs = model(data)
        loss    = criterion(outputs, lbs)
        loss.backward()
        optimizer.step()
        
        train_epoch_loss += loss.item()
    
    with torch.no_grad() :
        model.eval()
        val_epoch_loss = 0.
        
        for X_val_batch, y_val_batch in TestLoader :
            X_val_batch = X_val_batch.to(device)
            y_val_batch = y_val_batch.to(device)
            
            y_val_pred = model(X_val_batch)
            
            val_loss = criterion(y_val_pred, y_val_batch)
            
            val_epoch_loss += val_loss.item()
    
    avg_train_loss = float(train_epoch_loss) / len(TrainLoader)
    avg_val_loss   = float(val_epoch_loss) / len(TestLoader)
    scheduler.step(avg_val_loss)
    
    print(f'Epoch {epoch+1}: | Train Loss: {avg_train_loss:.4e} | Test Loss: {avg_val_loss:.4e}')
    
    if avg_val_loss < best_val_loss :
        print("Saving Model Checkpoint......")
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), model_checkpoint)

print("Training Complete!")
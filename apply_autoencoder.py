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

from models import *
from torch.utils.data import DataLoader, TensorDataset
from utilities import *
from attacks import *

os.environ["CUDA_VISIBLE_DEVICES"] = "8"
# EPS for FGSM attack
curr_attack = "FGSM_Linfinity"
eps_values = np.array([0.001, 0.002, 0.003, 0.005, 0.007, 0.010, 0.012, 0.017, 0.022, 
                        0.027, 0.030])
#eps_values = np.array([0.000, 0.001, 0.002, 0.003, 0.005, 0.007, 
#                       0.010, 0.020, 0.030])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Current Device: ", device)
print(torch.cuda.get_device_name(torch.cuda.current_device()))

filename = "RML2016.10b.dat"
snrs, mods, X, labels = process_data(filename)

x_train, x_test, y_train, y_test, test_labels, test_idx = train_test_split(X, labels, mods)

train_dataset = TensorDataset(x_train, y_train.type(torch.LongTensor))
test_dataset  = TensorDataset(x_test,  y_test.type(torch.LongTensor))
batch_size = 256
TrainLoader = DataLoader(train_dataset, batch_size = batch_size, 
                         shuffle = False)

model = CNN(input_size = 128)
model.load_state_dict(torch.load("CNN_base.pt"))
model = model.to(device)
encoder = AUTOENCODER()
encoder.load_state_dict(torch.load("AUTOENCODER.pt"))
encoder.to(device)
criterion  = nn.MSELoss()

T = 0
T_array = []

eps_values_T = np.array([0.001, 0.003, 0.005, 0.008, 0.012])
for eps in eps_values_T:
    for batch_idx, (data, lbs) in enumerate(TrainLoader) :
        data = data.to(device)
        lbs = lbs.to(device)
        #FGSM L-infinity Adversarial Retraining
        adv_data = FGSM_Linf_attack(model, device, data, lbs, eps)

        adv_data_pred = encoder(adv_data)
        loss = criterion(adv_data_pred,adv_data)
        T = max(T,loss)
    T_array.append(T)
print(T_array)
    

batch_size = 4
TestLoader  = DataLoader(test_dataset,  batch_size = batch_size, 
                         shuffle = False)
encoder.eval()
dtct_rate = []
#T = torch.mean(torch.stack(T_array))

for i in range(len(eps_values)) :
    detect_num = np.zeros(len(T_array))
    all_num = 0
    for X_val_batch, y_val_batch in TestLoader :
        X_val_batch = X_val_batch.to(device)
        y_val_batch = y_val_batch.to(device)
        
        adv_x = FGSM_Linf_attack(model, device, X_val_batch, y_val_batch, eps_values[i])
        adv_x_pred = encoder(adv_x)
        val_loss = criterion(adv_x_pred, adv_x)

        for m in range(len(T_array)):
            if val_loss >= T_array[m]:
                detect_num[m] += 1
        all_num += 1
        
    d_sum = np.sum(detect_num)
    detection_rate = float(d_sum) / (len(T_array)*float(all_num))  
    print(f'eps value {eps_values[i]}: Detection rate: {detection_rate:.3f}')   
    dtct_rate.append(detection_rate)

print(dtct_rate)

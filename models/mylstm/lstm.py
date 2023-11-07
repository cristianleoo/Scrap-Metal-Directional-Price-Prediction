import torch
from torch import nn as nn
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import os
import pandas as pd
import glob
import datetime

class LstmCell(nn.Module):
    def __init__(self, 
                 input_size, 
                 hidden_size, 
                 output_size, 
                 n_layers=1, # change number of layers to 3
                 dropout=0.0,
                 initialization='pytorch'
                 ):
        super(LstmCell, self).__init__()
        self.input_size= input_size
        self.hidden_size = hidden_size
        self.output_size= output_size
        self.n_layers = n_layers
        self.dropout_value = dropout

        # stack three LSTMs with 128, 64, and 32 nodes
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=self.hidden_size, num_layers=n_layers, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=self.hidden_size, hidden_size=128, num_layers=n_layers, batch_first=True)
        self.lstm3 = nn.LSTM(input_size=128, hidden_size=64, num_layers=n_layers, batch_first=True)

        self.dropout = nn.Dropout(dropout)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.linear = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
        self.flatten = nn.Flatten()
        self.fc_1 =  nn.Linear(self.hidden_size, 32) #fully connected 1

        # Parameters initalization
        if initialization=='Xavier':
            for param in self.lstm1.parameters():
                if len(param)>=2:
                    # init.xavier_uniform_()
                    init.xavier_uniform_(param.data)
        elif initialization=='Kaiming':
            for param in self.lstm1.parameters():
                if len(param)>=2:
                    init.kaiming_uniform_(param.data)
        
        # Initalize hook to store gradients
        self.gradients = [None] * len(list(self.lstm1.parameters()) + list(self.lstm2.parameters()) + list(self.lstm3.parameters()))
        self.register_hooks()

    def save_gradient(self, grad, idx):
        self.gradients[idx] = grad.clone

    def register_hooks(self):
        for idx, param in enumerate(list(self.lstm1.parameters()) + list(self.lstm2.parameters()) + list(self.lstm3.parameters())):
            param.register_hook(lambda grad, idx=idx: self.save_gradient(grad, idx))

    def forward(self, x):

        # lstm_out, (h_n, c_n) = self.lstm1(x)
        # lstm_out = self.dropout(lstm_out)
        # logits = self.linear(h_n[-1])
        
        lstm_out, (h_n, c_n) = self.lstm1(x)
        lstm_out = self.dropout(lstm_out)
        # lstm_out, (h_n, c_n) = self.lstm2(lstm_out)
        # lstm_out = self.dropout(lstm_out)
        # lstm_out, (h_n, c_n) = self.lstm3(lstm_out)
        # lstm_out = self.dropout(lstm_out)
        logits = self.linear(h_n[-1])
        probs = self.sigmoid(logits)
        # logits = self.relu(logits)
        # logits = self.linear(logits)
    
        return probs

    # def backward(self, x, y, criterion, optimizer):
    #     outputs = self.forward(x)
    #     loss = criterion(outputs, y)
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    #     return loss
    
    def get_gradients(self):
        return self.gradients
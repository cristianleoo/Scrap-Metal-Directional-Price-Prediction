import torch
from torch import nn as nn
from torch.nn import init
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
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=64, num_layers=n_layers, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=64, num_layers=n_layers, batch_first=True)
        self.lstm3 = nn.LSTM(input_size=64, hidden_size=32, num_layers=n_layers, batch_first=True)

        self.dropout = nn.Dropout(dropout)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.linear = nn.Linear(64, 1)
        self.softmax = nn.Softmax()
        self.flatten = nn.Flatten()

        # Parameters initalization
        if initialization=='Xavier':
            for param in self.lestm.parameters():
                if len(param)>=2:
                    init.xavier_uniform_()
        elif initialization=='Kaiming':
            for param in self.lstm.parameters():
                if len(param)>=2:
                    init.kaiming_uniform_()
        
        # Initalize hook to store gradients
        self.gradients = [None] * len(list(self.lstm1.parameters()) + list(self.lstm2.parameters()) + list(self.lstm3.parameters()))
        self.register_hooks()

    def save_gradient(self, grad, idx):
        self.gradients[idx] = grad.clone

    def register_hooks(self):
        for idx, param in enumerate(list(self.lstm1.parameters()) + list(self.lstm2.parameters()) + list(self.lstm3.parameters())):
            param.register_hook(lambda grad, idx=idx: self.save_gradient(grad, idx))

    def forward(self, x):
        h0 = torch.zeros(self.n_layers, x.size(0), 64).to(x.device)
        c0 = torch.zeros(self.n_layers, x.size(0), 64).to(x.device)
        out, _ = self.lstm1(x, (h0, c0))
        
        out = self.relu(out)
        out = self.dropout(out)

        # h1 = torch.zeros(1, x.size(0), 64).to(x.device)
        # c1 = torch.zeros(1, x.size(0), 64).to(x.device)
        # out, _ = self.lstm2(out, (h1, c1))

        # out = self.relu(out)
        # out = self.dropout(out)

        # h2 = torch.zeros(1, x.size(0), 32).to(x.device)
        # c2 = torch.zeros(1, x.size(0), 32).to(x.device)
        # out, _ = self.lstm3(out, (h2, c2))

        # out = self.relu(out)
        # out = self.dropout(out)
        out = self.flatten(out)

        out = self.linear(out)
        # out = self.linear(out)

        # out = self.softmax(out)
        
        return out

    def backward(self, x, y, criterion, optimizer):
        outputs = self.forward(x)
        loss = criterion(outputs, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss
    
    def get_gradients(self):
        return self.gradients
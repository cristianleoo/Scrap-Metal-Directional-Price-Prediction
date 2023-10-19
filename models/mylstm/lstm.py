import torch
from torch import nn
from torch.nn import init as init

class LstmCell(nn.Module):
    def __init__(self, 
                 input_size, 
                 hidden_size, 
                 output_size=1, 
                 n_layers=1,
                 dropout=0.0,
                 initialization='pytorch'
                 ):
        super(LstmCell, self).__init__()
        self.input_size= input_size
        self.hidden_size = hidden_size
        self.output_size= output_size
        self.n_layers = n_layers
        self.dropout_value = dropout

        self.lstm = nn.LSTM(input_size, hidden_size, output_size, n_layers, batch_first=True)
        self.dropout = nn.Dropout()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.linear = nn.Linear(hidden_size, output_size)
        self.batchnorm = nn.BatchNorm1d()

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
        self.gradients = [None] * len(list(self.lstm.parameters()))
        self.register_hooks()

    def save_gradient(self, grad, idx):
        self.gradients[idx] = grad.clone

    def register_hooks(self):
        for idx, param in enumerate(self.lstm.parameters()):
            param.register_hook(lambda grad, idx=idx: self.save_gradient(grad, idx))

    def forward(self, x):
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_size).to(x.device)
        x = self.lstm(x)
        out, _ = self.lstm(x, (h0, c0))
        out = self.linear(x[:, -1, :])
        out = self.dropout(self.dropout_value)
        return out

import torch
from torch import nn
from torch.nn import init as init
from torch.autograd import Variable
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class LstmCell(nn.Module):
    def __init__(self, 
                 input_size, 
                 hidden_size, 
                 output_size, 
                 n_layers=2,
                 dropout=0.0,
                 initialization='pytorch'
                 ):
        super(LstmCell, self).__init__()
        self.input_size= input_size
        self.hidden_size = hidden_size
        self.output_size= output_size
        self.n_layers = n_layers
        self.dropout_value = dropout

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=n_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.linear = nn.Linear(hidden_size, output_size)
        # self.batchnorm = nn.BatchNorm1d()

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
        # x = self.lstm(x)
        out, _ = self.lstm(x, (h0, c0))
        out = self.linear(out[:, -1, :])
        out = self.dropout(out)
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
    

class Trainer():
    def __init__(self):
        self.df = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.model = None

        try:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
            print(f'Using {self.device}')
        except Exception:
            self.device = torch.device('cpu')
            print(f'Using {self.device}')


    def get_data(self):
        df = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data/preprocessed/data.csv'))
        self.df = df
        return df
    
    def split(self, test_size=0.2):
        if self.df is None:
            df = self.get_data()
        else:
            df = self.df

        X = df.drop(['date', 'target'], axis=1)
        y = df['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=test_size, shuffle=False)
        
        X_train = X_train.values.reshape(X_train.shape[0], 1, X_train.shape[1])
        X_val = X_val.values.reshape(X_val.shape[0], 1, X_val.shape[1])
        X_test = X_test.values.reshape(X_test.shape[0], 1, X_test.shape[1])
        y_train = y_train.values.reshape(-1, 1)
        y_val = y_val.values.reshape(-1, 1)
        y_test = y_test.values.reshape(-1, 1)

        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test

        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def accuracy(self, y_true, y_pred):
        return accuracy_score(y_true, y_pred)
    
    def train(self, epochs=1000, early_stopping=50):
        if self.X_train is None:
            X_train, X_val, X_test, y_train, y_val, y_test = self.split()
        else:
            X_train = self.X_train
            X_val = self.X_val
            X_test = self.X_test
            y_train = self.y_train
            y_val = self.y_val
            y_test = self.y_test

        model = LstmCell(input_size=self.X_train.shape[-1], hidden_size=128, output_size=1, n_layers=2, dropout=0.2)
        model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        X_val = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        X_test = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        y_train = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        y_val = torch.tensor(y_val, dtype=torch.float32).to(self.device)
        y_test = torch.tensor(y_test, dtype=torch.float32).to(self.device)

        losses = { 'train': [], 'val': []}

        best_val_loss = float('inf')

        for epoch in range(epochs):
            outputs = model.forward(X_train)
            loss = criterion(outputs, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses['train'].append(loss.item())

            model.eval()
            val_outputs = model.forward(X_val)
            val_loss = criterion(val_outputs, y_val)

            losses['train'].append(loss.item())
            losses['val'].append(val_loss.item())

            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                self.save_model(model, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models/mylstm/lstm.pth'))
            
            print(f'Epoch: {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Best Val Loss: {best_val_loss:.4f}')
            if len(losses['val']) > early_stopping:
                if losses['val'][-1] > losses['val'][-early_stopping]:
                    print('Early stopping')
                    break
        
        self.model = model
        return model
    
    def test(self):
        if self.model is None:
            self.train()
        else:
            X_test = torch.tensor(self.X_test, dtype=torch.float32).to(self.device)
            y_test = torch.tensor(self.y_test, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            outputs = model.forward(X_test)
            criterion = nn.MSELoss()
            loss = criterion(outputs, y_test)
            print(f'Test loss: {loss.item():.4f}')
        return loss.item()
    
    def predict(self, X=None):
        if self.model is None:
            print('Model is not trained yet. Training now...')
            self.train()

        if isinstance(X, str):
            if X=='train':
                X = self.X_train
            elif X=='eval':
                X = self.X_val
            elif X=='test':
                X = self.X_test

        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.model.forward(X)

        return outputs
    
    def save_model(self, model, path):
        torch.save(model.state_dict(), path)
    
    def load_model(self, model, path):
        model.load_state_dict(torch.load(path))
        return model

model = Trainer()
model.train()
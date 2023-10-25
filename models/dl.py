import tqdm
import copy
import torch
from torch import nn
from torch.nn import init as init
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import RobustScaler, Normalizer, OneHotEncoder
from mylstm.lstm import LstmCell
from models.ingest import Ingest
    
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y
    
class DL(Ingest):
    def __init__(self):
        self.model = None

        # try:
        #     self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
        #     print(f'Using {self.device}')
        # except Exception:
        #     self.device = torch.device('cpu')
        #     print(f'Using {self.device}')
        self.device = torch.device('cpu')
    
    def train(self, epochs=200, early_stopping=50, n_layers=3):
        if self.X_train is None:
            X_train, X_val, X_test, y_train, y_val, y_test = self.split(dl=True)
        
        X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        X_val = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
        X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
        y_train = y_train.reshape(y_train.shape[0], 1, y_train.shape[1])
        y_val = y_val.reshape(y_val.shape[0], 1, y_val.shape[1])
        y_test = y_test.reshape(y_test.shape[0], 1, y_test.shape[1])

        model = LstmCell(input_size=self.X_train.shape[-1], hidden_size=128, output_size=3, n_layers=1, dropout=0.25)
        
        model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-5)
        criterion = nn.MSELoss()

        X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        X_val = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        X_test = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        y_train = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        y_val = torch.tensor(y_val, dtype=torch.float32).to(self.device)
        y_test = torch.tensor(y_test, dtype=torch.float32).to(self.device)

        # Set batch sizes
        batch_size = 5
        batches_per_epoch = len(X_train) #// batch_size

        performance = {'train_loss': [], 'val_loss': [],
                  'train_acc': [], 'val_acc': []}

        best_val_loss = float('inf')
        stop = False

        # Gradient Clipping
        max_grad_norm = 1.0  # You can adjust this value
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        self.model = model

        best_rmse = np.inf
        best_weights = None

        # training loop
        for epoch in range(batches_per_epoch):
            epoch_loss = []
            epoch_rmse = []
            # set model in training mode and run through each batch
            model.train()
            with tqdm.trange(batch_size, unit="batch", mininterval=0) as bar:
                bar.set_description(f"Epoch {epoch}")
                for i in bar:
                    # take a batch
                    start = i * batch_size
                    X_batch = X_train[start:start+batch_size]
                    y_batch = y_train[start:start+batch_size]
                    # forward pass
                    y_pred = model(X_batch)
                    loss = criterion(y_pred, y_batch)
                    # backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    # update weights
                    optimizer.step()
                    # compute and store metrics
                    # acc = (torch.argmax(y_pred, 1) == torch.argmax(y_batch, 1)).float().mean()
                    rmse = torch.sqrt(loss)
                    epoch_loss.append(float(loss))
                    epoch_rmse.append(float(rmse))
                    bar.set_postfix(
                        loss=float(loss),
                        rmse=float(rmse)
                    )
            # set model in evaluation mode and run through the test set
            model.eval()
            y_pred = model(X_test)
            y_test = y_test
            ce = criterion(y_pred, y_test)
            # acc = (torch.argmax(y_pred, 1) == torch.argmax(y_test, 1)).float().mean()
            rmse = float(torch.sqrt(ce))
            ce = float(ce)
            # acc = float(acc)
            performance['train_loss'].append(np.mean(epoch_loss))
            performance['train_acc'].append(np.mean(epoch_rmse))
            performance['val_loss'].append(ce)
            performance['val_acc'].append(rmse)
            if rmse < best_rmse:
                best_rmse = rmse
                best_weights = copy.deepcopy(model.state_dict())
            print(f"Epoch {epoch} Val Loss: {ce:.2f}, Rmse={rmse:.4f}")

        # Restore best model
        model.load_state_dict(best_weights)

            # if val_accuracy > best_val_accuracy:
            #     best_val_accuracy = val_accuracy
        # for epoch in range(epochs):
        #     model.train()
        #     for batch in train_dataloader:
        #         inputs, labels = batch
        #         outputs = model.forward(inputs)
        #         loss = criterion(outputs, labels.squeeze())
        #         optimizer.zero_grad()
        #         loss.backward()
        #         optimizer.step()
        #         losses['train'].append(loss.item())
        #         torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        #     model.eval()
        #     for batch in val_dataloader:
        #         inputs, labels = batch
                
        #         val_outputs = model.forward(inputs)
        #         val_loss = criterion(val_outputs, labels.squeeze())

        #         losses['train'].append(loss.item())
        #         losses['val'].append(val_loss.item())

        #         if val_loss.item() < best_val_loss:
        #             best_val_loss = val_loss.item()
        #             self.save_model(model, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models/mylstm/lstm.pth'))
                
        #         print(f'Epoch: {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Best Val Loss: {best_val_loss:.4f}')
        #         if len(losses['val']) > early_stopping:
        #             if losses['val'][-1] < losses['val'][-early_stopping]:
        #                 print('Early stopping')
        #                 stop = True
        #                 break
        #     if stop:
        #         break
        
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

model = DL()
model.train()
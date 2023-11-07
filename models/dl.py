import tqdm
import copy
import torch
from torch import nn
from torch.nn import init as init
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from mylstm.lstm import LstmCell
from ingest import Ingest
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
    
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
    def __init__(self, model_name='LSTM'):
        super().__init__()
        self.model = None
        self.model_name = model_name
        self.performance = None
        self.initialization = False

        # try:
        #     self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
        #     print(f'Using {self.device}')
        # except Exception:
        #     self.device = torch.device('cpu')
        #     print(f'Using {self.device}')
        self.device = torch.device('cpu')

    #############################
    # def loss(self, X, y):
    #     return F.binary_cross_entropy_with_logits(X, y)
        # return nn.BCELoss()(X, y)

    
    
    
    def prepare_data(self):
        if self.X_train is None:
            self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = self.split(dl=True)
        
        X_train, X_val, X_test, y_train, y_val, y_test = self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test
        X_train = X_train.values.reshape(X_train.shape[0], 1, X_train.shape[1])
        X_val = X_val.values.reshape(X_val.shape[0], 1, X_val.shape[1])
        X_test = X_test.values.reshape(X_test.shape[0], 1, X_test.shape[1])
        y_train = y_train.values.reshape(-1, 1)
        y_val = y_val.values.reshape(-1, 1)
        y_test = y_test.values.reshape(-1, 1)
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    #############################
    
    def tensorize(self, X_train, X_val, X_test, y_train, y_val, y_test):
        X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        X_val = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        X_test = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        y_train = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        y_val = torch.tensor(y_val, dtype=torch.float32).to(self.device)
        y_test = torch.tensor(y_test, dtype=torch.float32).to(self.device)

        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = X_train, X_val, X_test, y_train, y_val, y_test

        return X_train, X_val, X_test, y_train, y_val, y_test
    
    #############################

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            # torch.nn.init.xavier_uniform(m.weight)
            if self.initialization=='Xavier':
                init.xavier_uniform_(m.weight.data)
            elif self.initialization=='Kaiming':
                init.kaiming_uniform_(m.weight.data)
            m.bias.data.fill_(0.01)

    #############################
    
    def train(self, 
              epochs=100, 
              batch_size=32,
              hidden_size=64,
              early_stopping=50, 
              n_layers=1,
              lr=1e-3,
              weight_decay=1e-5,
              lstm_dropout=0.2,
              initialization=False):
        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_data()

        model = LstmCell(
            input_size=self.X_train.shape[-1], 
            hidden_size=hidden_size, 
            output_size=1, 
            n_layers=n_layers, 
            dropout=lstm_dropout,
            )
        
        
        if initialization:
            initialization = self.initialization
            model.apply(self.init_weights)
        model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True, eps=1e-8, betas=(0.9, 0.999), weight_decay=weight_decay)
        # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        criterion = nn.BCELoss()
        # criterion = nn.BCEWithLogitsLoss()

        X_train, X_val, X_test, y_train, y_val, y_test = self.tensorize(X_train, X_val, X_test, y_train, y_val, y_test)

        # Set batch sizes
        batch_size = batch_size
        n_epochs = epochs
        batch_start = torch.arange(0, len(X_train), batch_size)

        performance = {'train_loss': [], 'val_loss': [],
                  'train_acc': [], 'val_acc': []}

        # best_val_loss = float('inf')
        # stop = False

        # Gradient Clipping
        max_grad_norm = 1.0  # You can adjust this value
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        self.model = model

        best_acc = - np.inf
        best_loss = np.inf
        best_weights = None

        # training loop
        for epoch in range(n_epochs):
            epoch_loss = []
            epoch_acc = []
            # set model in training mode and run through each batch
            model.train()
            with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
                bar.set_description(f"Epoch {epoch}")
                for start in bar:
                    # take a batch
                    # start = i * batch_size
                    X_batch = X_train[start:start+batch_size]
                    y_batch = y_train[start:start+batch_size]
                    # forward pass
                    y_pred = model(X_batch)
                    # print(y_pred)
                    loss = criterion(y_pred, y_batch)
                    # backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    # update weights
                    optimizer.step()
                    acc = (y_pred.round() == y_batch).float().mean()
                    epoch_loss.append(float(loss))
                    epoch_acc.append(float(acc))
                    bar.set_postfix(
                        loss=float(loss),
                        acc=float(acc)
                    )
                
            # append average loss and accuracy over an epoch
            performance['train_loss'].append(np.mean(epoch_loss[:-1]))
            performance['train_acc'].append(np.mean(epoch_acc[:-1]))
            acc = np.mean(epoch_acc[:-1])

            # set model in evaluation mode and run through the test set
            model.eval()
            y_pred = model(X_val)
            val_loss = criterion(y_pred, y_val).item()
            val_acc = float((y_pred.round() == y_val).float().mean())
            performance['val_loss'].append(val_loss)
            performance['val_acc'].append(val_acc)
            if acc > best_acc:
                best_acc = acc
            # if loss < best_loss:
            #     best_loss = loss
                best_weights = copy.deepcopy(model.state_dict())
            print(f"Epoch {epoch} | Train Accuracy={acc:.2%} | Train Loss={loss:.4f} | Val Accuracy={val_acc:.2%} | Val Loss: {val_loss:.4f}")

        # Restore best model
        model.load_state_dict(best_weights)
        
        self.model = model
        self.performance = performance
        return model
    
    #############################
    
    def test(self):
        if self.model is None:
            self.train()
        else:
            X_test = torch.tensor(self.X_test, dtype=torch.float32).to(self.device)
            y_test = torch.tensor(self.y_test, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            outputs = self.model.forward(X_test)
            # criterion = nn.MSELoss()
            loss = self.loss(outputs, y_test)
            print(f'Test loss: {loss.item():.4f}')
        return loss.item()
    
    #############################
    
    def predict(self, X=None):
        if self.model is None:
            print('Model is not trained yet. Training now...')
            self.train()

        if isinstance(X, str):
            if X=='train':
                code = 'Train'
                X = self.X_train
                y = self.y_train
            elif X=='val':
                code = 'Validation'
                X = self.X_val
                y = self.y_val
            elif X=='test':
                code = 'Test'
                X = self.X_test
                y = self.y_test

        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.model.forward(X)
        
        # loss = self.loss(outputs, y)
        acc = (outputs.round() == y).float().mean()
        acc = float(acc)
        print(f'{code} Accuracy: {acc:.4f}')

        self.save_loss(acc, code, self.model_name)

        return outputs
    
    #############################
    
    def save_model(self, model, path):
        torch.save(model.state_dict(), path)

    #############################
    
    def load_model(self, model, path):
        model.load_state_dict(torch.load(path))
        return model
    
    #############################

    # create a function to plot the gradients per layer
    def plot_grad_flow(self):
        grad_flow = {}
        for n, t in list(enumerate([x.squeeze().detach().numpy() for x in self.get_gradients() if x is not None])):
            grad_flow[f'layer_{n+1}'] = []
            try:
                for i, l in enumerate(t):
                    grad_flow[f'layer_{n+1}'].append(l.mean())
            except:
                grad_flow[f'layer_{n+1}'].append(t.mean())

        fig, axs = plt.subplots(len(grad_flow.keys()), 1, figsize=(12, 4 * len(grad_flow.keys())))
        for n, layer in enumerate(grad_flow.keys()):
            axs[n].plot(grad_flow[layer], label=layer)
            axs[n].legend()
            axs[n].set_title(f'Gradient flow for {layer}')
        plt.legend()
        plt.show()

    #############################

    def get_gradients(self):
        gradients = []
        for i, param in enumerate(list(self.model.parameters())):
            gradients.append(param.grad)
        return gradients
    
    #############################

    def get_weights(self):
        weights = []
        for i, param in enumerate(list(self.model.parameters())):
            weights.append(param)
        return weights
    
    #############################

    def get_structure(self):
        structure = []
        for i, param in enumerate(list(self.model.parameters())):
            structure.append(param.shape)
        return structure

# model = DL()
# # model.train()
# model.predict('train')
# model.predict('val')
# model.predict('test')

import os
import json
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, Normalizer
from sklearn.model_selection import train_test_split

class Ingest():
    def __init__(self):
        self.df = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None

    #############################

    def get_data(self):
        df = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/preprocessed/data.csv'))
        self.df = df
        return df
    
    #############################
    
    def preprocess(self, train, val, test):
        scale = Normalizer()
        train = scale.fit_transform(train)
        val = scale.transform(val)
        test = scale.transform(test)
        return train, val, test
    
    #############################
    
    def one_hot_encode(self, y):
        enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        y = enc.fit_transform(y)
        return y
    
    #############################
    
    def split(self, test_size=0.1, dl=False):
        if self.df is None:
            df = self.get_data()
        else:
            df = self.df

        X = df.drop(['date', 'Target'], axis=1)
        y = df[['Target']]
        # if dl:
        #     y = self.one_hot_encode(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=test_size, shuffle=False)
        
        X_train, X_val, X_test = self.preprocess(X_train, X_val, X_test)

        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test

        return X_train, X_val, X_test, y_train, y_val, y_test
    
    #############################
    
    def save_loss(self, loss, code, model_name):
        with open("models/losses.json", 'r') as f:
            # Read file
            data = json.load(f)
        
        with open("models/losses.json", 'w') as f:
            # Append data
            data[model_name][code].append(loss)
            # Write file
            json.dump(data, f, indent=4)
        print('Loss saved to loss.json')
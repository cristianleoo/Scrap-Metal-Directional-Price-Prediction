import pandas as pd
import numpy as np
from ingest import Ingest
from mlmodels.myxgb import MyXGB

    
class ML(Ingest):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = None

    def train(self):
        if self.X_train is None:
            self.split()
        
        X_train, X_val, X_test, y_train, y_val, y_test = self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test
        
        model = MyXGB()
        model.grid_search(X_train, y_train, X_val, y_val, X_test, y_test)
        self.model = model
    
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
        
        y_pred = self.model.predict(X)
        return y_pred
    
    def feature_importance(self):
        if self.model is None:
            print('Model is not trained yet. Training now...')
            self.train()
        return self.model.feature_importances_

model = ML()
model.train()
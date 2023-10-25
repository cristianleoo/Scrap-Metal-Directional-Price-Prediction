from ingest import Ingest
from mlmodels.myxgb import MyXGB
from mlmodels.myrf import MyRF
from mlmodels.mylgb import MyLGB
from sklearn.metrics import mean_squared_error

    
class ML(Ingest):
    def __init__(self, model_name='XGB', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = model_name
        self.model = None

    #############################

    def train(self):
        if self.X_train is None:
            self.split()
        
        if self.model_name=='XGB':
            self.model = MyXGB()
        elif self.model_name=='Random Forest':
            self.model = MyRF()
        elif self.model_name=='LGB':
            self.model = MyLGB()
        X_train, X_val, X_test, y_train, y_val, y_test = self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test
        self.model.grid_search(X_train, y_train, X_val, y_val, X_test, y_test)

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
        
        y_pred = self.model.predict(X)
        mse = mean_squared_error(y, y_pred)
        print(f'{code} MSE: {mse:.4f}')

        self.save_loss(mse.round(4), code, self.model_name)

        return y_pred
    
    #############################
    
    def feature_importance(self):
        if self.model is None:
            print('Model is not trained yet. Training now...')
            self.train()
        return self.model.feature_importances_

model = ML(model_name='LGB')
# model.train()
model.predict('train')
model.predict('val')
model.predict('test')
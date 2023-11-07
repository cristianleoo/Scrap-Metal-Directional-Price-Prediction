from ingest import Ingest
from mlmodels.myxgb import MyXGB
from mlmodels.myrf import MyRF
from mlmodels.mylgb import MyLGB
from mlmodels.mysvc import MySVC
from mlmodels.mylogreg import MyLogReg
from mlmodels.mynb import MyNB
from mlmodels.mycatboost import MyCatBoost

    
class ML(Ingest):
    def __init__(self, model_name='XGB', binary=False, search='random', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = model_name
        self.model = None
        self.binary = binary
        self.search = search

    #############################

    def train(self):
        if self.X_train is None:
            self.split()
        
        if self.model_name=='XGB':
            self.model = MyXGB(binary=self.binary)
        elif self.model_name=='Random Forest':
            self.model = MyRF(binary=self.binary)
        elif self.model_name=='LGB':
            self.model = MyLGB(binary=self.binary, search=self.search)
        elif self.model_name=='SVC':
            self.model = MySVC(search=self.search)
        elif self.model_name=='Logistic Regression':
            self.model = MyLogReg(search=self.search)
        elif self.model_name=='Naive Bayes':
            self.model = MyNB(search=self.search)
        elif self.model_name=='CatBoost':
            self.model = MyCatBoost(search=self.search)
        X_train, X_val, X_test, y_train, y_val, y_test = self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test
        print(f'Training {self.model_name}...')
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
        
        loss, y_pred = self.model.loss(X, y)
        self.save_loss(loss.round(4), code, self.model_name)

        return y_pred
    
    #############################
    
    def feature_importance(self):
        if self.model is None:
            print('Model is not trained yet. Training now...')
            self.train()
        return self.model.feature_importances_

model = ML(model_name='CatBoost', binary=True, search='random')
# model.train()
model.predict('train')
model.predict('val')
model.predict('test')
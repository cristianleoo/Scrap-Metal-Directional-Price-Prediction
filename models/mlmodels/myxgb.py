from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, accuracy_score
import xgboost as xgb

class MyXGB:
    def __init__(self, params=None, binary=True):
        if params is None:
            self.params = {
                # 'max_depth': 3,
                'learning_rate': 0.1,
                'n_estimators': 1000,
                # 'objective': 'reg:squarederror',
                'booster': 'gbtree',
                'n_jobs': -1,
                'random_state': 42,
            }
            if binary:
                self.params['objective'] = 'binary:logistic'
            else:
                self.params['objective'] = 'reg:squarederror'
        else:
            self.params = params
        self.model = None

        self.param_grid = {
            # 'max_depth': [3, 5, 7],
            'learning_rate': [0.001, 0.01, 0.1],
            # 'num_boost_round': [100, 500, 1000],
            'booster': ['gbtree'],
            'n_jobs': [-1],
            'random_state': [42],
        }
        self.num_boost_round = 1000
        if binary:
            self.param_grid['objective'] = ['binary:logistic']
            # use cross entropy for binary classification
            self.scoring = 'neg_log_loss'
        else:
            self.param_grid['objective'] = ['reg:squarederror']
            self.scoring = 'neg_root_mean_squared_error'
        
        self.binary = binary

    #############################

    def fit(self, X_train, y_train, X_val=None, y_val=None, X_test=None, y_test=None):
        dtrain = xgb.DMatrix(X_train, label=y_train, silent=False)
        watchlist = [(dtrain, 'train')]
        evals_result = {}
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            watchlist.append((dval, 'validation'))
        if X_test is not None and y_test is not None:
            dtest = xgb.DMatrix(X_test, label=y_test)
            watchlist.append((dtest, 'test'))
        self.model = xgb.train(self.params, dtrain, num_boost_round=self.num_boost_round, evals=watchlist,
                               evals_result=evals_result, early_stopping_rounds=100, verbose_eval=False)
        return evals_result
    
    #############################

    def predict(self, X):
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)
    
    #############################

    def loss(self, X, y):
        y_pred = self.predict(X)
        if self.binary:
            y_pred[y_pred>0.5] = 1
            y_pred[y_pred<=0.5] = 0
            return accuracy_score(y, y_pred), y_pred
        else:

            return mean_squared_error(y, self.predict(X)), y_pred
        
    #############################

    def grid_search(self, X_train, y_train, X_val=None, y_val=None, X_test=None, y_test=None, cv=5):
        if self.binary:
            estimator = xgb.XGBClassifier(**self.params)
        else:
            estimator = xgb.XGBRegressor(**self.params)


        grid_search = GridSearchCV(estimator=estimator, 
                                   param_grid=self.param_grid, 
                                   cv=cv,
                                   scoring=self.scoring, 
                                   refit=True, 
                                   n_jobs=-1, 
                                   verbose=1, 
                                   return_train_score=True)
        grid_search.fit(X_train, y_train)
        self.params = grid_search.best_params_
        self.model = grid_search.best_estimator_
        print(grid_search.best_params_)

        # Fit best model and print rmse for train, val, test
        self.fit(X_train, y_train, X_val, y_val, X_test, y_test)
        loss, _ = self.loss(X_train, y_train)
        print(f"Train Loss: {loss:.4f}")
        if X_val is not None and y_val is not None:
            loss, _ = self.loss(X_val, y_val)
            print(f"Val Loss: {loss:.4f}")
        if X_test is not None and y_test is not None:
            loss, _ = self.loss(X_test, y_test)
            print(f"Test Loss: {loss:.4f}")
        return grid_search
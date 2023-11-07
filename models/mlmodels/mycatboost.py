from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

class MyCatBoost:
    def __init__(self, params=None, search='grid'):
        if params is None:
            self.params = {
                'iterations': 500,
                'learning_rate': 0.1,
                'depth': 6,
                'loss_function': 'Logloss',
                'eval_metric': 'Accuracy',
                'random_seed': 42,
                'bagging_temperature': 0.75,
                'od_type': 'Iter',
                'od_wait': 100,
                'verbose': False
            }
        else:
            self.params = params
        self.model = None

        self.param_grid = {
            'iterations': [500, 1000, 1500],
            'learning_rate': [0.01, 0.1, 0.5],
            'depth': [4, 6, 8],
            'loss_function': ['Logloss', 'CrossEntropy'],
            'eval_metric': ['Accuracy'],
            'random_seed': [42],
            'bagging_temperature': [0.5, 0.75, 1],
            'od_type': ['Iter', 'IncToDec', 'Round'],
            'od_wait': [100, 200, 300],
            'verbose': [False]
        }
        self.search = search

    #############################

    def fit(self, X_train, y_train, X_val=None, y_val=None, X_test=None, y_test=None):
        self.model = CatBoostClassifier(**self.params)
        self.model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
        return self.model
    
    #############################

    def predict(self, X):
        return self.model.predict(X)
    
    #############################
    
    def loss(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred), y_pred

    #############################

    def grid_search(self, X_train, y_train, X_val=None, y_val=None, X_test=None, y_test=None, scoring='accuracy', params=None):
        if params is None:
            params = self.params
        estimator = CatBoostClassifier(**params)
        if self.search=='random':
            grid_search = RandomizedSearchCV(estimator=estimator, 
                                   param_distributions=self.param_grid, 
                                   scoring=scoring, 
                                   refit=True, 
                                   n_jobs=-1, 
                                   n_iter=10,
                                   verbose=1, 
                                   return_train_score=True)
        else:
            grid_search = GridSearchCV(estimator=estimator, 
                                    param_grid=self.param_grid, 
                                    scoring=scoring, 
                                    refit=True, 
                                    n_jobs=-1, 
                                    verbose=1, 
                                    return_train_score=True)
        grid_search.fit(X_train, y_train)
        self.params = grid_search.best_params_
        self.model = grid_search.best_estimator_
        print(grid_search.best_params_)

        # Fit best model and print accuracy for train, val, test
        self.fit(X_train, y_train, X_val, y_val, X_test, y_test)
        acc, y_pred = self.loss(X_train, y_train)
        print(f"Train Accuracy: {acc:.4f}")
        if X_val is not None and y_val is not None:
            acc, y_pred = self.loss(X_val, y_val)
            print(f"Val Accuracy: {acc:.4f}")
        if X_test is not None and y_test is not None:
            acc, y_pred = self.loss(X_test, y_test)
            print(f"Test Accuracy: {acc:.4f}")
        return grid_search

    #############################


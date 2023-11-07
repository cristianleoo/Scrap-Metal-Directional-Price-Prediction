from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score

class MySVC:
    def __init__(self, params=None, search='random'):
        if params is None:
            self.params = {
                'C': 1.0,
                'kernel': 'rbf',
                'degree': 3,
                'gamma': 'scale',
                'coef0': 0.0,
                'shrinking': True,
                'probability': False,
                'tol': 0.001,
                'cache_size': 200,
                'class_weight': None,
                'verbose': False,
                'max_iter': -1,
                'decision_function_shape': 'ovr',
                'break_ties': False,
                'random_state': None
            }
        else:
            self.params = params
        self.model = None

        self.param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'degree': [2, 3, 4],
            'gamma': ['scale', 'auto'],
            'coef0': [0.0, 1.0, 2.0],
            'shrinking': [True, False],
            'probability': [True, False],
            'tol': [0.001, 0.01, 0.1],
            'cache_size': [200, 500, 1000],
            'class_weight': [None, 'balanced'],
            'verbose': [False],
            'max_iter': [-1],
            'decision_function_shape': ['ovr', 'ovo'],
            'break_ties': [False],
            'random_state': [None]
        }
        self.search = search

    #############################

    def fit(self, X_train, y_train, X_val=None, y_val=None, X_test=None, y_test=None):
        self.model = SVC(**self.params)
        self.model.fit(X_train, y_train)
        return self.model
    
    #############################

    def predict(self, X):
        return self.model.predict(X)
    
    #############################
    
    def loss(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred), y_pred

    #############################

    def grid_search(self, X_train, y_train, X_val=None, y_val=None, X_test=None, y_test=None, cv=5, scoring='accuracy', params=None):
        y_train = y_train.values.ravel()
        y_val = y_val.values.ravel()
        y_test = y_test.values.ravel()

        if params is None:
            params = self.params
        estimator = SVC(**params)
        if self.search=='random':
            grid_search = RandomizedSearchCV(estimator=estimator, 
                                   param_distributions=self.param_grid, 
                                   cv=cv,
                                   scoring=scoring, 
                                   refit=True, 
                                   n_jobs=-1, 
                                   n_iter=10,
                                   verbose=2, 
                                   return_train_score=True)
        else:
            grid_search = GridSearchCV(estimator=estimator, 
                                    param_grid=self.param_grid, 
                                    cv=cv,
                                    scoring=scoring, 
                                    refit=True, 
                                    n_jobs=-1, 
                                    verbose=2, 
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

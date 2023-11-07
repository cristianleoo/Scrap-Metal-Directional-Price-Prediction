from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score

class MyLogReg:
    def __init__(self, params=None, search='grid'):
        if params is None:
            self.params = {
                'penalty': 'l2',
                'dual': False,
                'tol': 0.0001,
                'C': 1.0,
                'fit_intercept': True,
                'intercept_scaling': 1,
                'class_weight': None,
                'random_state': None,
                'solver': 'lbfgs',
                'max_iter': 100,
                'multi_class': 'auto',
                'verbose': 0,
                'warm_start': False,
                'n_jobs': None,
                'l1_ratio': None
            }
        else:
            self.params = params
        self.model = None

        self.param_grid = {
            'penalty': ['l1', 'l2', 'elasticnet', 'none'],
            'dual': [True, False],
            'tol': [0.0001, 0.001, 0.01],
            'C': [0.1, 1, 10],
            'fit_intercept': [True, False],
            'intercept_scaling': [0.1, 1, 10],
            'class_weight': [None, 'balanced'],
            'random_state': [None],
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
            'max_iter': [100, 200, 300],
            'multi_class': ['auto', 'ovr', 'multinomial'],
            'verbose': [0],
            'warm_start': [True, False],
            'n_jobs': [None],
            'l1_ratio': [None]
        }
        self.search = search

    #############################

    def fit(self, X_train, y_train, X_val=None, y_val=None, X_test=None, y_test=None):
        self.model = LogisticRegression(**self.params)
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
        if params is None:
            params = self.params
        estimator = LogisticRegression(**params)
        if self.search=='random':
            grid_search = RandomizedSearchCV(estimator=estimator, 
                                   param_distributions=self.param_grid, 
                                   cv=cv,
                                   scoring=scoring, 
                                   refit=True, 
                                   n_jobs=-1, 
                                   n_iter=10,
                                   verbose=1, 
                                   return_train_score=True)
        else:
            grid_search = GridSearchCV(estimator=estimator, 
                                    param_grid=self.param_grid, 
                                    cv=cv,
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

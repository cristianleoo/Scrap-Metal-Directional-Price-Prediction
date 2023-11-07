from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, accuracy_score

class MyRF:
    def __init__(self, params=None, binary=True, search='grid'):
        if params is None:
            self.params = {
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'max_features': 'auto',
                'bootstrap': True,
                'random_state': 42,
            }
        else:
            self.params = params
        self.model = None

        self.param_grid = {
            'n_estimators': [100, 200, 500],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt'],
            'bootstrap': [True, False],
            'random_state': [42],
        }
        self.binary = binary
        self.search = search

    #############################

    def fit(self, X_train, y_train, X_val=None, y_val=None, X_test=None, y_test=None):
        if self.binary:
            self.model = RandomForestClassifier(**self.params)
        else:
            self.model = RandomForestRegressor(**self.params)
        self.model.fit(X_train, y_train)
        return self.model
    
    #############################

    def predict(self, X):
        return self.model.predict(X)
    
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

    def grid_search(self, X_train, y_train, X_val=None, y_val=None, X_test=None, y_test=None, cv=5, scoring='neg_root_mean_squared_error', params=None):
        if params is None:
            params = self.params
        if self.binary:
            estimator = RandomForestClassifier(**params)
        else:
            estimator = RandomForestRegressor(**params)
        if self.search=='random':
            grid_search = RandomizedSearchCV(estimator=estimator, 
                                   param_distributions=self.param_grid, 
                                   cv=cv,
                                   scoring=scoring, 
                                   refit=True, 
                                   n_jobs=-1, 
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

        # Fit best model and print rmse for train, val, test
        self.fit(X_train, y_train, X_val, y_val, X_test, y_test)
        loss, y_pred = self.loss(X_train, y_train)
        print(f"Train Loss: {loss:.4f}")
        if X_val is not None and y_val is not None:
            loss, y_pred = self.loss(X_val, y_val)
            print(f"Val Loss: {loss:.4f}")
        if X_test is not None and y_test is not None:
            loss, y_pred = self.loss(X_test, y_test)
            print(f"Test Loss: {loss:.4f}")
        return grid_search


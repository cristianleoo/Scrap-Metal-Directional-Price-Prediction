from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

class MyRF:
    def __init__(self, params=None):
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

    def fit(self, X_train, y_train, X_val=None, y_val=None, X_test=None, y_test=None):
        self.model = RandomForestRegressor(**self.params)
        self.model.fit(X_train, y_train)
        return self.model

    def predict(self, X):
        return self.model.predict(X)

    def grid_search(self, X_train, y_train, X_val=None, y_val=None, X_test=None, y_test=None, cv=5, scoring='neg_root_mean_squared_error', params=None):
        if params is None:
            params = self.params
        grid_search = GridSearchCV(estimator=RandomForestRegressor(**params), 
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
        y_pred = self.predict(X_train)
        mse = mean_squared_error(y_train, y_pred)
        print(f"Train MSE: {mse:.4f}")
        if X_val is not None and y_val is not None:
            y_pred = self.predict(X_val)
            mse = mean_squared_error(y_val, y_pred)
            print(f"Val MSE: {mse:.4f}")
        if X_test is not None and y_test is not None:
            y_pred = self.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            print(f"Test MSE: {mse:.4f}")
        return grid_search

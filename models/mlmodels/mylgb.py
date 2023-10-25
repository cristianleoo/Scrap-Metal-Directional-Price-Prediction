from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

class MyLGB:
    def __init__(self, params=None):
        if params is None:
            self.params = {
                'n_estimators': 100,
                'max_depth': -1,
                'learning_rate': 0.1,
                'num_leaves': 31,
                'min_child_samples': 20,
                'subsample': 1.0,
                'colsample_bytree': 1.0,
                'reg_alpha': 0.0,
                'reg_lambda': 0.0,
                'random_state': 42,
            }
        else:
            self.params = params
        self.model = None

        self.param_grid = {
            'n_estimators': [100, 200, 500],
            'max_depth': [-1, 3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'num_leaves': [31, 50, 100],
            'min_child_samples': [20, 50, 100],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            'reg_alpha': [0.0, 0.1, 0.5],
            'reg_lambda': [0.0, 0.1, 0.5],
            'random_state': [42],
        }

    def fit(self, X_train, y_train, X_val=None, y_val=None, X_test=None, y_test=None):
        self.model = LGBMRegressor(**self.params)
        self.model.fit(X_train, y_train)
        return self.model

    def predict(self, X):
        return self.model.predict(X)

    def grid_search(self, X_train, y_train, X_val=None, y_val=None, X_test=None, y_test=None, cv=5, scoring='neg_root_mean_squared_error', params=None):
        if params is None:
            params = self.params
        grid_search = GridSearchCV(estimator=LGBMRegressor(**params), 
                                   param_grid=self.param_grid, 
                                   cv=cv,
                                   scoring=scoring, 
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

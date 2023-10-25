from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error
import xgboost as xgb

class MyXGB:
    def __init__(self, params=None):
        if params is None:
            self.params = {
                'max_depth': 3,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'objective': 'reg:squarederror',
                'booster': 'gbtree',
                'n_jobs': -1,
                'random_state': 42,
            }
        else:
            self.params = params
        self.model = None

        self.param_grid = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.001, 0.01, 0.1],
            'n_estimators': [100, 200, 500],
            'objective': ['reg:squarederror'],
            'booster': ['gbtree'],
            'n_jobs': [-1],
            'random_state': [42],
        }

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
        self.model = xgb.train(self.params, dtrain, num_boost_round=self.params['n_estimators'], evals=watchlist,
                               evals_result=evals_result, early_stopping_rounds=10, verbose_eval=False)
        return evals_result

    def predict(self, X):
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)

    def cv(self, X, y, cv=5, scoring='neg_root_mean_squared_error', params=None):
        if params is None:
            params = self.params
        dtrain = xgb.DMatrix(X, label=y)
        cv_results = xgb.cv(params, dtrain, num_boost_round=params['n_estimators'], nfold=cv, metrics=scoring,
                            early_stopping_rounds=10, seed=42)
        return cv_results

    def grid_search(self, X_train, y_train, X_val=None, y_val=None, X_test=None, y_test=None, cv=5, scoring='neg_root_mean_squared_error'):
        grid_search = GridSearchCV(estimator=xgb.XGBRegressor(**self.params), param_grid=self.param_grid, cv=cv,
                                   scoring=scoring, refit=True, n_jobs=-1, verbose=2, return_train_score=True)
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
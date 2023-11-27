import os
import json
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from fengineering import FeatureEngineering
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class Ingest():
    """
    Ingest class for data ingestion, preprocessing, and splitting.

    Methods:
        - get_data(): Read data from a CSV file.
        - scale(train, test): Scale numerical features in the train and test sets.
        - impute_numerical(train, test, y='Target'): Impute missing values in numerical features.
        - binarizer(y): Binarize the target variable.
        - split(test_size=0.2, val=False, transform=True, new_cols=None, drop_unimportant=False, scale=False):
            Split the data into train/validation/test sets and perform preprocessing.

    Example:
        ingest = Ingest()
        X_train, X_val, X_test, y_train, y_val, y_test = ingest.split(val=False, drop_unimportant=False, transform=True, scale=False)
    """
    def __init__(self):
        self.df = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.imputer = IterativeImputer(RandomForestRegressor(), max_iter=10, random_state=0)

    #############################

    def get_data(self):
        """
        Read data from a CSV file.

        Returns:
            - df: DataFrame, the loaded dataset
        """
        df = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/preprocessed/data.csv'))
        self.df = df
        return df
    
    #############################
    
    def scale(self, train, test):
        """
        Scale numerical features in the train and test sets.

        Parameters:
            - train: DataFrame, training set
            - test: DataFrame, test set

        Returns:
            - train, test: Scaled training and test sets
        """
        scale = MinMaxScaler() #StandardScaler() #RobustScaler()
        cols_to_scale = [col for col in train.columns if not col.endswith('_sentiment') or not col.endswith('_rating') or col != 'Target' or not col.endswith('benzinga')]
        
        train[cols_to_scale] = scale.fit_transform(train[cols_to_scale])
        test[cols_to_scale] = scale.transform(test[cols_to_scale])
        return train, test
    
    #############################

    def impute_numerical(self, train, test, y='Target'):
        """
        Impute missing values in numerical features.

        Parameters:
            - train: DataFrame, training set
            - test: DataFrame, test set
            - y: str, target variable name

        Returns:
            - train, test: Imputed training and test sets
        """
        # check if columns are numerical
        #numerical_cols = train.drop([y], axis=1).select_dtypes(include=['int64', 'float64']).columns
        cols = [col for col in train.columns[:-1] if not col.endswith('benzinga')] #train.drop([y], axis=1).columns
        train[cols] = self.imputer.fit_transform(train[cols])
        test[cols] = self.imputer.transform(test[cols])
        return train, test
    
    #############################

    def binarizer(self, y):
        """
        Binarize the target variable.

        Parameters:
            - y: array-like, target variable

        Returns:
            - binarized_y: array-like, binarized target variable
        """
        return np.where(y > 0, 1.0, 0.0)
    
    #############################
    
    def split(self, 
              test_size=0.2, 
              val=False, 
              transform=True,
              new_cols=None,
              drop_unimportant=False, 
              scale=False):
        """
        Split the data into train/validation/test sets and perform preprocessing.

        Parameters:
            - test_size: float, default=0.2, proportion of data for the test set
            - val: bool, default=False, whether to include a validation set
            - transform: bool, default=True, whether to perform feature engineering
            - new_cols: list, default=None, additional columns for feature engineering
            - drop_unimportant: bool, default=False, whether to drop unimportant features
            - scale: bool, default=False, whether to scale numerical features

        Returns:
            - X_train, X_val, X_test, y_train, y_val, y_test: Train/validation/test sets
        """
        if self.df is None:
            df = self.get_data()
        else:
            df = self.df

        df.drop(['date'], axis=1, inplace=True)
        # df['scrap_change'] = df['Target'].shift(-1)
        df['Target'] = self.binarizer(df['Target'])
        
        train = df.iloc[:int((1-test_size)*len(df)), :]
        test = df.iloc[int((1-test_size)*len(df)):, :]

        train, test = self.impute_numerical(train, test)
        if transform and new_cols is None:
            cols_to_transform = [col for col in train.columns[:-1] if not col.endswith('benzinga')]
            fe = FeatureEngineering(train, test, 'Target', cols_to_transform)
            train, test, self.new_cols = fe.main()
            train, test = fe.main_feature_elimination(train, test)
        elif transform and new_cols is not None:
            cols_to_transform = [col for col in train.columns[:-1] if not col.endswith('benzinga')]
            fe = FeatureEngineering(train, test, 'Target', new_cols)
            train, test, self.new_cols = fe.apply_arithmetic_operations(new_cols)
            train, test = fe.main_feature_elimination(train, test)

        test = test[train.columns]

        X_train = train.drop(['Target'], axis=1)
        y_train = train[['Target']].astype(int)
        X_test = test.drop(['Target'], axis=1)
        y_test = test[['Target']].astype(int)
        if transform:
            X_train, n_imp_features = fe.final_selection(X_train, y_train)
            X_test=X_test[n_imp_features]

        if scale:
            X_train, X_test = self.scale(X_train, X_test)

        if drop_unimportant:
            X_train.drop(fe.unimportant_features, axis=1, inplace=True)
            X_test.drop(fe.unimportant_features, axis=1, inplace=True)

        if val:
            X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=test_size, shuffle=False)
        else:
            X_val, y_val = None, None
        
        # X_train, X_val, X_test = self.preprocess(X_train, X_val, X_test)

        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test

        return X_train, X_val, X_test, y_train, y_val, y_test
    
    #############################
    
    def save_loss(self, loss, code, model_name):
        """
        Save loss to a JSON file.

        Parameters:
            - loss: float, loss value
            - code: str, model identifier
            - model_name: str, model name
        """
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models/losses.json')
        with open(path, 'r') as f:
            # Read file
            data = json.load(f)
        
        with open(path, 'w') as f:
            # Append data
            data[model_name][code].append(loss)
            # Write file
            json.dump(data, f, indent=4)
        print('Loss saved to loss.json')


ingest = Ingest()
X_train, X_val, X_test, y_train, y_val, y_test = ingest.split(val=False,
                                                               drop_unimportant=False, 
                                                               transform=True,
                                                            #    new_cols=new_cols,
                                                               scale=False
                                                               ) # drop_unimportant=False give the best result for CatBoost

X_train.head()
from prettytable import PrettyTable
import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer, FunctionTransformer, MinMaxScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

class FeatureEngineering:
    def __init__(self):
        self.unimportant_features=[]
        self.overall_best_score=0
        self.overall_best_col='none'
        # self.sc=MinMaxScaler()
        
    # def min_max_scaler(self, train, test, column):
    #     '''
    #     Min Max just based on train might have an issue if test has extreme values, hence changing the denominator uding overall min and max
    #     '''
    #     max_val=max(train[column].max(),test[column].max())
    #     min_val=min(train[column].min(),test[column].min())

    #     train[column]=(train[column]-min_val)/(max_val-min_val)
    #     test[column]=(test[column]-min_val)/(max_val-min_val)

    #     return train,test  

    def transformer(self, train, test, cont_cols, target):
        '''
        Algorithm applies multiples transformations on selected columns and finds the best transformation using a single variable model performance
        '''
        train_copy = train.copy()
        test_copy = test.copy()
        table = PrettyTable()
        table.field_names = ['Feature', 'Initial ROC_AUC', 'Transformation', 'Tranformed ROC_AUC']

        for col in cont_cols:

            for c in ["log_"+col, "sqrt_"+col, "bx_cx_"+col, "y_J_"+col, "log_sqrt"+col, "pow_"+col, "pow2_"+col]:
                if c in train_copy.columns:
                    train_copy = train_copy.drop(columns=[c])

            # Log Transformation after MinMax Scaling (keeps data between 0 and 1)
            train_copy["log_"+col] = np.log1p(train_copy[col])
            test_copy["log_"+col] = np.log1p(test_copy[col])

            # Square Root Transformation
            train_copy["sqrt_"+col] = np.sqrt(train_copy[col])
            test_copy["sqrt_"+col] = np.sqrt(test_copy[col])

            # Box-Cox transformation
            combined_data = pd.concat([train_copy[[col]], test_copy[[col]]], axis=0)
            epsilon = 1e-5
            transformer = PowerTransformer(method='box-cox')
            scaled_data = transformer.fit_transform(combined_data + epsilon)

            train_copy["bx_cx_" + col] = scaled_data[:train_copy.shape[0]]
            test_copy["bx_cx_" + col] = scaled_data[train_copy.shape[0]:]
            # Yeo-Johnson transformation
            transformer = PowerTransformer(method='yeo-johnson')
            train_copy["y_J_"+col] = transformer.fit_transform(train_copy[[col]])
            test_copy["y_J_"+col] = transformer.transform(test_copy[[col]])

            # Power transformation, 0.25
            power_transform = lambda x: np.power(x + 1 - np.min(x), 0.25)
            transformer = FunctionTransformer(power_transform)
            train_copy["pow_"+col] = transformer.fit_transform(train_copy[[col]])
            test_copy["pow_"+col] = transformer.transform(test_copy[[col]])

            # Power transformation, 2
            power_transform = lambda x: np.power(x + 1 - np.min(x), 2)
            transformer = FunctionTransformer(power_transform)
            train_copy["pow2_"+col] = transformer.fit_transform(train_copy[[col]])
            test_copy["pow2_"+col] = transformer.transform(test_copy[[col]])

            # Log to power transformation
            train_copy["log_sqrt"+col] = np.log1p(train_copy["sqrt_"+col])
            test_copy["log_sqrt"+col] = np.log1p(test_copy["sqrt_"+col])

            temp_cols = [col, "log_"+col, "sqrt_"+col, "bx_cx_"+col, "y_J_"+col,  "pow_"+col , "pow2_"+col,"log_sqrt"+col]

            pca = TruncatedSVD(n_components=1)
            x_pca_train = pca.fit_transform(train_copy[temp_cols])
            x_pca_test = pca.transform(test_copy[temp_cols])
            x_pca_train = pd.DataFrame(x_pca_train, columns=[col+"_pca_comb"])
            x_pca_test = pd.DataFrame(x_pca_test, columns=[col+"_pca_comb"])
            temp_cols.append(col+"_pca_comb")

            test_copy = test_copy.reset_index(drop=True)

            train_copy = pd.concat([train_copy, x_pca_train], axis='columns')
            test_copy = pd.concat([test_copy, x_pca_test], axis='columns')

            kf = KFold(n_splits=5, shuffle=False)

            auc_scores = []

            for f in temp_cols:
                X = train_copy[[f]].values
                y = train_copy[target].values

                auc = []
                for train_idx, val_idx in kf.split(X, y):
                    X_train, y_train = X[train_idx], y[train_idx]
                    x_val, y_val = X[val_idx], y[val_idx]
                    model =   SVC(gamma="auto", probability=True, random_state=42)
                    # model =  RandomForestClassifier(n_estimators=100, random_state=42)
                    # model =   LogisticRegression() # since it is a large dataset, Logistic Regression would be a good option to save time
                    model.fit(X_train, y_train)
                    y_pred = model.predict_proba(x_val)[:,1]
                    auc.append(roc_auc_score(y_val, y_pred))
                auc_scores.append((f, np.mean(auc)))

                if self.overall_best_score < np.mean(auc):
                    self.overall_best_score = np.mean(auc)
                    self.overall_best_col = f

                if f == col:
                    orig_auc = np.mean(auc)

            best_col, best_auc = sorted(auc_scores, key=lambda x: x[1], reverse=True)[0]
            cols_to_drop = [f for f in temp_cols if f != best_col]
            final_selection = [f for f in temp_cols if f not in cols_to_drop]

            if cols_to_drop:
                self.unimportant_features = self.unimportant_features+cols_to_drop
            table.add_row([col,orig_auc,best_col ,best_auc])
        print(table)   
        print("overall best CV ROC AUC score: ",self.overall_best_score)
        return train_copy, test_copy

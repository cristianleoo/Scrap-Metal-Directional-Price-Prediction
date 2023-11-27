from prettytable import PrettyTable
import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer, FunctionTransformer, StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from tqdm import tqdm
import xgboost as xgb
from catboost import CatBoostClassifier
import lightgbm as lgb
import seaborn as sns
import matplotlib.pyplot as plt

class FeatureEngineering:
    def __init__(self, train, test, target, cont_cols):
        """
        Initialize the FeatureEngineering class.

        Parameters:
        - train (pd.DataFrame): Training dataset.
        - test (pd.DataFrame): Testing dataset.
        - target (str): Name of the target column.
        - cont_cols (list): List of continuous feature columns.
        """
        self.unimportant_features=[]
        self.overall_best_score=0
        self.overall_best_col='none'
        self.train=train
        self.test=test
        self.target=target
        self.cont_cols=cont_cols
        self.cat_cols = train.drop(cont_cols, axis=1).columns[:-1] #these are the text features
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

    def transformer(self):
        """
        Apply multiple transformations on selected columns and find the best transformation using a single variable model performance.
        """
        train_copy = self.train.copy()
        test_copy = self.test.copy()
        table = PrettyTable()
        table.field_names = ['Feature', 'Initial ROC_AUC', 'Transformation', 'Tranformed ROC_AUC']

        for col in self.cont_cols:

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
                y = train_copy[self.target].values

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
        self.train = train_copy
        self.test = test_copy

    #############################

    def numerical_clustering(self):
        """
        Cluster numerical features based on unimportant features and calculate ROC AUC score on the transformed data.
        """
        table = PrettyTable()
        table.field_names = ['Clustered Feature', 'ROC AUC (CV-TRAIN)']
        for col in self.cont_cols:
            sub_set=[f for f in self.unimportant_features if col in f]
            temp_train=self.train[sub_set]
            temp_test=self.test[sub_set]
            sc=StandardScaler()
            temp_train=sc.fit_transform(temp_train)
            temp_test=sc.transform(temp_test)
            model = KMeans()

            # print(ideal_clusters)
            kmeans = KMeans(n_clusters=10)
            kmeans.fit(np.array(temp_train))
            labels_train = kmeans.labels_

            self.train[col+"_unimp_cluster_WOE"] = labels_train
            self.test[col+"_unimp_cluster_WOE"] = kmeans.predict(np.array(temp_test))

            
            kf=KFold(n_splits=5, shuffle=False)
            
            X=self.train[[col+"_unimp_cluster_WOE"]].values
            y=self.train[self.target].astype(int).values

            auc=[]
            for train_idx, val_idx in kf.split(X,y):
                X_train,y_train=X[train_idx],y[train_idx]
                x_val,y_val=X[val_idx],y[val_idx]
                model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.02, max_depth=6, random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict_proba(x_val)[:,1]
                auc.append(roc_auc_score(y_val,y_pred))
                
            table.add_row([col+"_unimp_cluster_WOE",np.mean(auc)])
            if overall_best_score<np.mean(auc):
                overall_best_score=np.mean(auc)
                overall_best_col=col+"_unimp_cluster_WOE"

        print(table)
        print("overall best CV score: ", overall_best_score)

    #############################

    def better_features(self, cols, best_score):
        """
        Generate new columns by applying arithmetic operations on existing ones.

        Parameters:
        - cols (list): List of feature columns.
        - best_score (float): Best ROC AUC score obtained so far.

        Returns:
        - pd.DataFrame, pd.DataFrame, list: Updated train dataset, test dataset, and list of new columns.
        """
        new_cols = []
        skf = KFold(n_splits=5, shuffle=True, random_state=42)  # Stratified k-fold object
        best_list=[]
        for i in tqdm(range(len(cols)), desc='Generating Columns'):
            col1 = cols[i]
            temp_df = pd.DataFrame()  # Temporary dataframe to store the generated columns
            temp_df_test = pd.DataFrame()  # Temporary dataframe for test data

            for j in range(i+1, len(cols)):
                col2 = cols[j]
                # Multiply
                temp_df[col1 + '*' + col2] = self.train[col1] * self.train[col2]
                temp_df_test[col1 + '*' + col2] = self.test[col1] * self.test[col2]

                # Divide (col1 / col2)
                temp_df[col1 + '/' + col2] = self.train[col1] / (self.train[col2] + 1e-5)
                temp_df_test[col1 + '/' + col2] = self.test[col1] / (self.test[col2] + 1e-5)

                # Divide (col2 / col1)
                temp_df[col2 + '/' + col1] = self.train[col2] / (self.train[col1] + 1e-5)
                temp_df_test[col2 + '/' + col1] = self.test[col2] / (self.test[col1] + 1e-5)

                # Subtract
                temp_df[col1 + '-' + col2] = self.train[col1] - self.train[col2]
                temp_df_test[col1 + '-' + col2] = self.test[col1] - self.test[col2]

                # Add
                temp_df[col1 + '+' + col2] = self.train[col1] + self.train[col2]
                temp_df_test[col1 + '+' + col2] = self.test[col1] + self.test[col2]

            SCORES = []
            for column in temp_df.columns:
                scores = []
                for train_index, val_index in skf.split(self.train, self.train[self.target]):
                    X_train, X_val = temp_df[column].iloc[train_index].values.reshape(-1, 1), temp_df[column].iloc[val_index].values.reshape(-1, 1)
                    y_train, y_val = self.train[self.target].astype(int).iloc[train_index], self.train[self.target].astype(int).iloc[val_index]
                    model = SVC(probability=True)#HistGradientBoostingClassifier(max_iter=300, learning_rate=0.02, max_depth=6, random_state=42)
                    model.fit(X_train, y_train)
                    y_pred = model.predict_proba(X_val)[:,1]
                    score = roc_auc_score( y_val, y_pred)
                    scores.append(score)
                mean_score = np.mean(scores)
                SCORES.append((column, mean_score))

            if SCORES:
                best_col, best_auc = sorted(SCORES, key=lambda x: x[1],reverse=True)[0]
                corr_with_other_cols = self.train.drop([self.target] + new_cols, axis=1).corrwith(temp_df[best_col])
                if (corr_with_other_cols.abs().max() < 0.9 or best_auc > best_score) and corr_with_other_cols.abs().max() !=1 :
                    self.train[best_col] = temp_df[best_col]
                    self.test[best_col] = temp_df_test[best_col]
                    new_cols.append(best_col)
                    print(f"Added column '{best_col}' with ROC AUC Score: {best_auc:.4f} & Correlation {corr_with_other_cols.abs().max():.4f}")

        return self.train, self.test, new_cols
    
    #############################

    def main(self):
        """
        Perform feature generation and selection using the better_features method.

        Returns:
        - pd.DataFrame, pd.DataFrame, list: Updated train dataset, test dataset, and list of new columns.
        """
        selected_features=[f for f in self.train.columns if self.train[f].nunique()>2 and f not in self.unimportant_features]
        selected_features = [col for col in selected_features if col not in self.cat_cols]
        train, test, new_cols= self.better_features(selected_features, self.overall_best_score)
        return train, test, new_cols
    
    #############################
    def apply_arithmetic_operations(self, expressions_list):
        """
        Apply arithmetic operations on selected feature columns.

        Parameters:
        - expressions_list (list): List of arithmetic expressions.

        Returns:
        - pd.DataFrame, pd.DataFrame: Updated train and test datasets.
        """
        train_df = self.train.copy()
        test_df = self.test.copy()

        for expression in expressions_list:
            if expression not in train_df.columns:
                # Split the expression based on operators (+, -, *, /)
                parts = expression.split('+') if '+' in expression else \
                        expression.split('-') if '-' in expression else \
                        expression.split('*') if '*' in expression else \
                        expression.split('/')

                # Get the DataFrame column names involved in the operation
                cols = [col for col in parts]

                # Perform the corresponding arithmetic operation based on the operator in the expression
                if cols[0] in train_df.columns and cols[1] in train_df.columns:
                    if '+' in expression:
                        train_df[expression] = train_df[cols[0]] + train_df[cols[1]]
                        test_df[expression] = test_df[cols[0]] + test_df[cols[1]]
                    elif '-' in expression:
                        train_df[expression] = train_df[cols[0]] - train_df[cols[1]]
                        test_df[expression] = test_df[cols[0]] - test_df[cols[1]]
                    elif '*' in expression:
                        train_df[expression] = train_df[cols[0]] * train_df[cols[1]]
                        test_df[expression] = test_df[cols[0]] * test_df[cols[1]]
                    elif '/' in expression:
                        train_df[expression] = train_df[cols[0]] / (train_df[cols[1]]+1e-5)
                        test_df[expression] = test_df[cols[0]] /( test_df[cols[1]]+1e-5)
        
        return train_df, test_df
    
    #############################
    # FEATURE SELECTION

    def feature_elimination(self, train, test):
        """
        Eliminate unimportant features based on correlation and clustering.

        Parameters:
        - train (pd.DataFrame): Training dataset.
        - test (pd.DataFrame): Testing dataset.

        Returns:
        - pd.DataFrame, pd.DataFrame: Updated train and test datasets.
        """
        first_drop=[ f for f in self.unimportant_features if f in train.columns]
        train=train.drop(columns=first_drop)
        test=test.drop(columns=first_drop)
        final_drop_list=[]

        table = PrettyTable()
        table.field_names = ['Original', 'Final Transformation', 'ROV AUC CV']
        threshold=0.95
        # It is possible that multiple parent features share same child features, so store selected features to avoid selecting the same feature again
        best_cols=[]

        for col in self.cont_cols:
            sub_set=[f for f in train.drop(self.cat_cols, axis=1).columns if (str(col) in str(f)) and (train[f].nunique()>2)]
        #     print(sub_set)
            if len(sub_set)>2:
                correlated_features = []

                for i, feature in enumerate(sub_set):
                    # Check correlation with all remaining features
                    for j in range(i+1, len(sub_set)):
                        correlation = np.abs(train[feature].corr(train[sub_set[j]]))
                        # If correlation is greater than threshold, add to list of highly correlated features
                        if correlation > threshold:
                            correlated_features.append(sub_set[j])

                # Remove duplicate features from the list
                correlated_features = list(set(correlated_features))
        #         print(correlated_features)
                if len(correlated_features)>=2:
                    temp_train=train[correlated_features]
                    temp_test=test[correlated_features]
                    #Scale before applying PCA
                    sc=StandardScaler()
                    temp_train=sc.fit_transform(temp_train)
                    temp_test=sc.transform(temp_test)

                    # Initiate PCA
                    pca=TruncatedSVD(n_components=1)
                    x_pca_train=pca.fit_transform(temp_train)
                    x_pca_test=pca.transform(temp_test)
                    x_pca_train=pd.DataFrame(x_pca_train, columns=[col+"_pca_comb_final"])
                    # x_pca_test=pd.DataFrame(x_pca_test, columns=[col+"_pca_comb_final"])
                    train=pd.concat([train,x_pca_train], axis=1)
                    # test=pd.concat([test,x_pca_test], axis=1)
                    test[col+"_pca_comb_final"]=x_pca_test

                    # Clustering
                    model = KMeans()
                    kmeans = KMeans(n_clusters=10)
                    kmeans.fit(np.array(temp_train))
                    labels_train = kmeans.labels_

                    train[col+'_final_cluster'] = labels_train
                    test[col+'_final_cluster'] = kmeans.predict(np.array(temp_test))


                    correlated_features=correlated_features+[col+"_pca_comb_final",col+"_final_cluster"]

                    # See which transformation along with the original is giving you the best univariate fit with target
                    kf=KFold(n_splits=5, shuffle=True, random_state=42)

                    scores=[]

                    for f in correlated_features:
                        X=train[[f]].values
                        y=train[self.target].astype(int).values

                        auc=[]
                        for train_idx, val_idx in kf.split(X,y):
                            X_train,y_train=X[train_idx],y[train_idx]
                            X_val,y_val=X[val_idx],y[val_idx]

                            model = HistGradientBoostingClassifier (max_iter=300, learning_rate=0.02, max_depth=6, random_state=42)
                            model.fit(X_train,y_train)
                            y_pred = model.predict_proba(X_val)[:,1]
                            score = roc_auc_score( y_val, y_pred)
                            auc.append(score)
                        if f not in best_cols:
                            scores.append((f,np.mean(auc)))
                    best_col, best_auc=sorted(scores, key=lambda x:x[1], reverse=True)[0]
                    best_cols.append(best_col)

                    cols_to_drop = [f for f in correlated_features if  f not in best_cols]
                    if cols_to_drop:
                        final_drop_list=final_drop_list+cols_to_drop
                    table.add_row([col,best_col ,best_auc])

        print(table)
        return train, test

    #############################

    def scaling(self, train, test):
        """
        Scale selected features in the train and test datasets.

        Parameters:
        - train (pd.DataFrame): Training dataset.
        - test (pd.DataFrame): Testing dataset.

        Returns:
        - pd.DataFrame, pd.DataFrame: Scaled train and test datasets.
        """
        final_features=[f for f in train.columns if f not in [self.target] and f not in self.cat_cols]
        final_features=[*set(final_features)]

        sc=StandardScaler()

        train_scaled=train.copy()
        test_scaled=test.copy()
        train_scaled[final_features]=sc.fit_transform(train[final_features])
        test_scaled[final_features]=sc.transform(test[final_features])
        return train_scaled, test_scaled
    
    #############################

    def post_processor(self, train, test):
        """
        Remove duplicate features after scaling.

        Parameters:
        - train (pd.DataFrame): Training dataset.
        - test (pd.DataFrame): Testing dataset.

        Returns:
        - pd.DataFrame, pd.DataFrame: Updated train and test datasets.
        """
        # train, test = self.scaling(train, test)
        '''
        After Scaleing, some of the features may be the same and can be eliminated
        '''
        cols=[f for f in train.columns if self.target not in f and "OHE" not in f and f not in self.cat_cols]
        train_cop=train.copy()
        test_cop=test.copy()
        drop_cols=[]
        for i, feature in enumerate(cols):
            for j in range(i+1, len(cols)):
                if sum(abs(train_cop[feature]-train_cop[cols[j]]))==0:
                    if cols[j] not in drop_cols:
                        drop_cols.append(cols[j])
        print(drop_cols)
        train_cop.drop(columns=drop_cols,inplace=True)
        test_cop.drop(columns=drop_cols,inplace=True)
        
        return train_cop, test_cop
    
    #############################

    def main_feature_elimination(self, train, test):
        """
        Perform feature elimination, scaling, and post-processing.

        Parameters:
        - train (pd.DataFrame): Training dataset.
        - test (pd.DataFrame): Testing dataset.

        Returns:
        - pd.DataFrame, pd.DataFrame: Updated train and test datasets.
        """
        train, test = self.scaling(train, test)
        train, test = self.feature_elimination(train, test)
        train, test = self.post_processor(train, test)
        return train, test

    #############################

    def get_most_important_features(self,
                                    X_train, 
                                    y_train, 
                                    n, 
                                    model_input,
                                    visualize=True):
        """
        Get the most important features using a specified model.

        Parameters:
        - X_train (pd.DataFrame): Training feature dataset.
        - y_train (pd.Series): Training target variable.
        - n (int): Number of top features to select.
        - model_input (str): Model name ('cat', 'xgb', or 'lgbm').
        - visualize (bool): Whether to visualize feature importances.

        Returns:
        - list: Top N important feature names.
        """
        xgb_params = {
                'n_jobs': -1,
                'eval_metric': 'logloss',
                'objective': 'binary:logistic',
                'tree_method': 'hist',
                'verbosity': 0,
                'random_state': 42,
            }
        lgb_params = {
                'objective': 'binary',
                'metric': 'logloss',
                'boosting_type': 'gbdt',
                'random_state': 42,
            }
        cb_params = {
                'grow_policy': 'Depthwise',
                'bootstrap_type': 'Bayesian',
                'od_type': 'Iter',
                'eval_metric': 'AUC',
                'loss_function': 'Logloss',
                'random_state': 42,
            }
        if 'xgb' in model_input:
            model = xgb.XGBClassifier(**xgb_params)
        elif 'cat' in model_input:
            model=CatBoostClassifier(**cb_params)
        else:
            model=lgb.LGBMClassifier(**lgb_params)
            
        X_train.drop(self.cat_cols, axis=1, inplace=True)

        X_train_fold, X_val_fold = X_train.iloc[:int(len(X_train)*0.8), :], X_train.iloc[int(len(X_train)*0.8):, :]
        y_train_fold, y_val_fold = y_train.iloc[:int(len(X_train)*0.8)], y_train.iloc[int(len(X_train)*0.8):]

        model.fit(X_train_fold, y_train_fold)
        
        y_pred = model.predict_proba(X_val_fold)[:,1]
        auc_score = roc_auc_score(y_val_fold, y_pred)
        feature_importances = model.feature_importances_

        feature_importance_list = [(X_train.columns[i], importance) for i, importance in enumerate(feature_importances)]
        sorted_features = sorted(feature_importance_list, key=lambda x: x[1], reverse=True)
        top_n_features = [feature[0] for feature in sorted_features[:n]]

        display_features=top_n_features[:10]
        
        if visualize:
            sns.set_palette("Set2")
            plt.figure(figsize=(8, 6))
            plt.barh(range(len(display_features)), [feature_importances[X_train.columns.get_loc(feature)] for feature in display_features])
            plt.yticks(range(len(display_features)), display_features, fontsize=12)
            plt.xlabel('Feature Importance', fontsize=14)
            plt.ylabel('Features', fontsize=10)
            plt.title(f'Top {10} of {n} Feature Importances with ROC AUC score {auc_score}', fontsize=16)
            plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature on top
            plt.grid(axis='x', linestyle='--', alpha=0.7)
            plt.xticks(fontsize=8)
            plt.yticks(fontsize=8)

            # Add data labels on the bars
            for index, value in enumerate([feature_importances[X_train.columns.get_loc(feature)] for feature in display_features]):
                plt.text(value + 0.005, index, f'{value:.3f}', fontsize=12, va='center')

            plt.tight_layout()
            plt.show()

        return top_n_features
    
    #############################

    def final_selection(self, X_train, y_train, n=150, visualize=True):
        """
        Perform final feature selection using different models.

        Parameters:
        - X_train (pd.DataFrame): Training feature dataset.
        - y_train (pd.Series): Training target variable.
        - n (int): Number of top features to select.
        - visualize (bool): Whether to visualize feature importances.

        Returns:
        - pd.DataFrame, list: Updated train dataset and list of selected features.
        """
        n_imp_features_cat=self.get_most_important_features(X_train.reset_index(drop=True), y_train, n, 'cat', visualize)
        n_imp_features_xgb=self.get_most_important_features(X_train.reset_index(drop=True), y_train, n, 'xgb', visualize)
        n_imp_features_lgbm=self.get_most_important_features(X_train.reset_index(drop=True), y_train, n, 'lgbm', visualize)
        n_imp_features=[*set(n_imp_features_xgb+n_imp_features_lgbm+n_imp_features_cat)]
        n_imp_features.extend(self.cat_cols)
        print(f"{len(n_imp_features)} features have been selected from three algorithms for the final model")
        X_train=X_train[n_imp_features]
        return X_train, n_imp_features
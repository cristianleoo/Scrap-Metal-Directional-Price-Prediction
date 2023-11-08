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

class FeatureEngineering:
    def __init__(self, train, test, target, cont_cols):
        self.unimportant_features=[]
        self.overall_best_score=0
        self.overall_best_col='none'
        self.train=train
        self.test=test
        self.target=target
        self.cont_cols=cont_cols
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
        '''
        Algorithm applies multiples transformations on selected columns and finds the best transformation using a single variable model performance
        '''
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
            y=self.train["defects"].astype(int).values

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
                    model = SVC()#HistGradientBoostingClassifier(max_iter=300, learning_rate=0.02, max_depth=6, random_state=42)
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

        #return train, test, new_cols
    
    def main(self):
        selected_features=[f for f in self.train.columns if self.train[f].nunique()>2 and f not in self.unimportant_features]
        train, test, new_cols= self.better_features(selected_features, self.overall_best_score)
        return train, test, new_cols
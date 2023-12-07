import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
# import kfold
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score, f1_score, precision_score, recall_score
import numpy as np
import optuna
import optuna.visualization as vis
from functools import partial
import gc
from copy import deepcopy
import pandas as pd
import os
import json
import pprint
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


class Splitter:
    """
    Splitter class for generating train/validation splits.

    Parameters:
        - test_size: float, default=0.2, the proportion of the dataset to include in the validation split
        - kfold: bool, default=False, whether to use KFold cross-validation
        - n_splits: int, default=5, number of folds in KFold

    Methods:
        - split_data(X, y, random_state_list): Splits the data into train/validation sets based on the specified configuration.
    """
    def __init__(self, test_size=0.2, kfold=False, n_splits=5):
        self.test_size = test_size
        self.kfold = kfold
        self.n_splits = n_splits

    def split_data(self, X, y, random_state_list):
        if self.kfold:
            for random_state in random_state_list:
                kf = KFold(n_splits=self.n_splits, random_state=random_state, shuffle=True)
                for train_index, val_index in kf.split(X, y):
                    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
                    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
                    yield X_train, X_val, y_train, y_val

############################################

class Classifier:
    """
    Classifier class for defining and initializing multiple classification models.

    Parameters:
        - n_estimators: int, default=100, number of estimators for ensemble models
        - device: str, default="cpu", device type for CatBoost ("cpu" or "gpu")
        - random_state: int, default=0, random state for reproducibility

    Methods:
        - _define_model(): Define and return a dictionary of classification models.
    """
    def __init__(self, n_estimators=100, device="cpu", random_state=0):
        self.n_estimators = n_estimators
        self.device = device
        self.random_state = random_state
        self.models = self._define_model()
        self.len_models = len(self.models)
        
    def _define_model(self):
        """
        Define and return a dictionary of classification models.

        Returns:
            - models: dictionary, collection of classification models
        """
        xgb_params = {
            'n_estimators': self.n_estimators,
            'learning_rate': 0.1,
            'max_depth': 4,
            'subsample': 0.8,
            'colsample_bytree': 0.1,
            'n_jobs': -1,
            'eval_metric': 'logloss',
            'objective': 'binary:logistic',
            'tree_method': 'hist',
            'verbosity': 0,
            'random_state': self.random_state,
#             'class_weight':class_weights_dict,
        }
        if self.device == 'gpu':
            xgb_params['tree_method'] = 'gpu_hist'
            xgb_params['predictor'] = 'gpu_predictor'
            
        xgb_params2=xgb_params.copy() 
        xgb_params2['subsample']= 0.3
        xgb_params2['max_depth']=8
        xgb_params2['learning_rate']=0.005
        xgb_params2['colsample_bytree']=0.9

        xgb_params3=xgb_params.copy() 
        xgb_params3['subsample']= 0.6
        xgb_params3['max_depth']=6
        xgb_params3['learning_rate']=0.02
        xgb_params3['colsample_bytree']=0.7      
        
        lgb_params = {
            'n_estimators': self.n_estimators,
            'max_depth': 8,
            'learning_rate': 0.02,
            'subsample': 0.20,
            'colsample_bytree': 0.56,
            'reg_alpha': 0.25,
            'reg_lambda': 5e-08,
            'objective': 'binary',
            'boosting_type': 'gbdt',
            'device': self.device,
            'random_state': self.random_state,
#             'class_weight':class_weights_dict,
        }
        lgb_params2 = {
            'n_estimators': self.n_estimators,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.20,
            'colsample_bytree': 0.56,
            'reg_alpha': 0.25,
            'reg_lambda': 5e-08,
            'objective': 'binary',
            'boosting_type': 'gbdt',
            'device': self.device,
            'random_state': self.random_state,
        }
        lgb_params3=lgb_params.copy()  
        lgb_params3['subsample']=0.9
        lgb_params3['reg_lambda']=0.3461495211744402
        lgb_params3['reg_alpha']=0.3095626288582237
        lgb_params3['max_depth']=8
        lgb_params3['learning_rate']=0.007
        lgb_params3['colsample_bytree']=0.5

        lgb_params4=lgb_params2.copy()  
        lgb_params4['subsample']=0.7
        lgb_params4['reg_lambda']=0.1
        lgb_params4['reg_alpha']=0.2
        lgb_params4['max_depth']=10
        lgb_params4['learning_rate']=0.007
        lgb_params4['colsample_bytree']=0.5
        cb_params = {
            'iterations': self.n_estimators,
            'depth': 6,
            'learning_rate': 0.1,
            'l2_leaf_reg': 0.7,
            'random_strength': 0.2,
            'max_bin': 200,
            'od_wait': 65,
            'one_hot_max_size': 120,
            'grow_policy': 'Depthwise',
            'bootstrap_type': 'Bayesian',
            'od_type': 'Iter',
            'eval_metric': 'AUC',
            'loss_function': 'Logloss',
            'task_type': self.device.upper(),
            'random_state': self.random_state,
        }
        cb_sym_params = cb_params.copy()
        cb_sym_params['grow_policy'] = 'SymmetricTree'
        cb_loss_params = cb_params.copy()
        cb_loss_params['grow_policy'] = 'Lossguide'
        
        cb_params2=  cb_params.copy()
        cb_params2['learning_rate']=0.01
        cb_params2['depth']=8
        
        cb_params3={
            'iterations': self.n_estimators,
            'random_strength': 0.1, 
            'one_hot_max_size': 70, 
            'max_bin': 100, 
            'learning_rate': 0.008, 
            'l2_leaf_reg': 0.3, 
            'grow_policy': 'Depthwise', 
            'depth': 10, 
            'max_bin': 200,
            'od_wait': 65,
            'bootstrap_type': 'Bayesian',
            'od_type': 'Iter',
            'eval_metric': 'AUC',
            'loss_function': 'Logloss',
            'task_type': self.device.upper(),
            'random_state': self.random_state,
        }
        cb_params4=  cb_params.copy()
        cb_params4['learning_rate']=0.01
        cb_params4['depth']=12
        dt_params= {'min_samples_split': 30, 'min_samples_leaf': 10, 'max_depth': 8, 'criterion': 'gini'}
        
        models = {
            'xgb': xgb.XGBClassifier(**xgb_params),
            'xgb2': xgb.XGBClassifier(**xgb_params2),
            'xgb3': xgb.XGBClassifier(**xgb_params3),
            'lgb': lgb.LGBMClassifier(**lgb_params),
            'lgb2': lgb.LGBMClassifier(**lgb_params2),
            'lgb3': lgb.LGBMClassifier(**lgb_params3),
            'lgb4': lgb.LGBMClassifier(**lgb_params4),
            'cat': CatBoostClassifier(**cb_params),
            'cat2': CatBoostClassifier(**cb_params2),
            'cat3': CatBoostClassifier(**cb_params2),
            'cat4': CatBoostClassifier(**cb_params2),
            "cat_sym": CatBoostClassifier(**cb_sym_params),
            "cat_loss": CatBoostClassifier(**cb_loss_params),
            'hist_gbm' : HistGradientBoostingClassifier (max_iter=300, learning_rate=0.001,  max_leaf_nodes=80,
                                                         max_depth=6,random_state=self.random_state),#class_weight=class_weights_dict, 
#             'gbdt': GradientBoostingClassifier(max_depth=6,  n_estimators=1000,random_state=self.random_state),
            'lr': LogisticRegression(),
            'rf': RandomForestClassifier(max_depth= 9,max_features= 'auto',min_samples_split= 10,
                                                          min_samples_leaf= 4,  n_estimators=500,random_state=self.random_state),
#             'svc': SVC(gamma="auto", probability=True),
#             'knn': KNeighborsClassifier(n_neighbors=5),
#             'mlp': MLPClassifier(random_state=self.random_state, max_iter=1000),
#              'etr':ExtraTreesClassifier(min_samples_split=55, min_samples_leaf= 15, max_depth=10,
#                                        n_estimators=200,random_state=self.random_state),
#             'dt' :DecisionTreeClassifier(**dt_params,random_state=self.random_state),
#             'ada': AdaBoostClassifier(random_state=self.random_state),
#             'ann':ann,
                                       
        }
        return models
    
############################################

class OptunaWeights:
    """
    OptunaWeights class for optimizing ensemble weights using Optuna.

    Parameters:
        - random_state: int, random state for reproducibility
        - n_trials: int, default=5000, number of trials for optimization

    Methods:
        - fit(y_true, y_preds): Optimize ensemble weights using Optuna.
        - predict(y_preds): Predict using the optimized ensemble weights.
        - fit_predict(y_true, y_preds): Fit and predict using the optimized ensemble weights.
        - weights(): Get the optimized ensemble weights.
    """
    def __init__(self, random_state, n_trials=5000):
        self.study = None
        self.weights = None
        self.random_state = random_state
        self.n_trials = n_trials

    def _objective(self, trial, y_true, y_preds):
        """
        Objective function for Optuna optimization.

        Parameters:
            - trial: Optuna trial
            - y_true: array-like, true labels
            - y_preds: list of array-like, predicted labels from each model

        Returns:
            - auc_score: float, ROC AUC score
        """
        # Define the weights for the predictions from each model
        weights = [trial.suggest_float(f"weight{n}", -2, 3) for n in range(len(y_preds))]

        # Calculate the weighted prediction
        weighted_pred = np.average(np.array(y_preds).T, axis=1, weights=weights)

        auc_score = roc_auc_score(y_true, weighted_pred)
        log_loss_score=log_loss(y_true, weighted_pred)
        return auc_score#/log_loss_score

    def fit(self, y_true, y_preds):
        """
        Optimize ensemble weights using Optuna.

        Parameters:
            - y_true: array-like, true labels
            - y_preds: list of array-like, predicted labels from each model
        """
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        sampler = optuna.samplers.CmaEsSampler(seed=self.random_state)
        pruner = optuna.pruners.HyperbandPruner()
        self.study = optuna.create_study(sampler=sampler, pruner=pruner, study_name="OptunaWeights", direction='maximize')
        objective_partial = partial(self._objective, y_true=y_true, y_preds=y_preds)
        self.study.optimize(objective_partial, n_trials=self.n_trials)
        self.weights = [self.study.best_params[f"weight{n}"] for n in range(len(y_preds))]

    def predict(self, y_preds):
        """
        Predict using the optimized ensemble weights.

        Parameters:
            - y_preds: list of array-like, predicted labels from each model

        Returns:
            - weighted_pred: array-like, weighted ensemble predictions
        """
        assert self.weights is not None, 'OptunaWeights error, must be fitted before predict'
        weighted_pred = np.average(np.array(y_preds).T, axis=1, weights=self.weights)
        return weighted_pred

    def fit_predict(self, y_true, y_preds):
        """
        Fit and predict using the optimized ensemble weights.

        Parameters:
            - y_true: array-like, true labels
            - y_preds: list of array-like, predicted labels from each model

        Returns:
            - weighted_pred: array-like, weighted ensemble predictions
        """
        self.fit(y_true, y_preds)
        return self.predict(y_preds)
    
    def weights(self):
        """
        Get the optimized ensemble weights.

        Returns:
            - weights: list, optimized ensemble weights
        """
        return self.weights
    
############################################

class Trainer():
    """
    Trainer class for training and evaluating an ensemble of classification models.

    Methods:
        - save_log(losses, X_train): Save training log to a JSON file.
        - main(X_train, X_test, y_train, y_test, best_ensemble=False): Train and evaluate the ensemble.

    Example:
        trainer = Trainer()
        trainer.main(X_train, X_test, y_train, y_test, best_ensemble=False)
    """
    def save_log(self, losses, X_train):
        """
        Save training log to a JSON file.

        Parameters:
            - losses: DataFrame, training losses
            - X_train: DataFrame, training features
        """
        models = losses['Model'].unique().tolist()
        models.pop(models.index('Ensemble'))
        train_acc = losses[losses['Model']=='Ensemble']['Train Acc'].values[0]
        acc = losses[losses['Model']=='Ensemble']['Test Acc'].values[0]
        roc_auc = losses[losses['Model']=='Ensemble']['Test ROC AUC'].values[0]
        # f1 = losses[losses['Model']=='Ensemble']['Test F1'].values[0]
        # precision = losses[losses['Model']=='Ensemble']['Test Precision'].values[0]
        # recall = losses[losses['Model']=='Ensemble']['Test Recall'].values[0]
        features = X_train.columns

        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models/logs.json')
        with open(path, 'r') as f:
            # Read file
            data = json.load(f)
        
        if data.keys().__len__() == 0:
            n = 1
        else:
            print(data.keys())
            n = max([int(key.split(' ')[1]) for key in data.keys()]) + 1
        
        with open(path, 'w') as f:
            # Append data
            data[f'Ensemble {n}'] = {
                'models': models,
                'train_acc': train_acc,
                'acc': acc,
                'roc_auc': roc_auc,
                # 'f1': f1,
                # 'precision': precision,
                # 'recall': recall,
                'features': features.tolist()
            }
            # data[f'Ensemble {n}']['models'].append(models)
            # data[f'Ensemble {n}']['train_acc'].append(train_acc)
            # data[f'Ensemble {n}']['acc'].append(acc)
            # data[f'Ensemble {n}']['roc_auc'].append(roc_auc)
            # data[f'Ensemble {n}']['f1'].append(f1)
            # data[f'Ensemble {n}']['precision'].append(precision)
            # data[f'Ensemble {n}']['recall'].append(recall)
            # data[f'Ensemble {n}']['features'].append(features)
            # Write file
            json.dump(data, f, indent=4)
    
    ############################################

    def main(self, X_train, X_test, y_train, y_test, best_ensemble=False):
        """
        Train and evaluate the ensemble.

        Parameters:
            - X_train: DataFrame, training features
            - X_test: DataFrame, test features
            - y_train: Series, training labels
            - y_test: Series, test labels
            - best_ensemble: bool, whether to use Optuna for optimizing ensemble weights
        """
        kfold = False
        n_splits = 1 if not kfold else 5
        random_state = 2023
        random_state_list = [42] # used by split_data [71]
        n_estimators = 9999
        early_stopping_rounds = 300
        verbose = False
        device = 'cpu'

        splitter = Splitter(kfold=kfold, n_splits=n_splits)

        # Initialize an array for storing test predictions
        test_predss = np.zeros(X_test.shape[0])
        ensemble_score = []
        weights = []
        trained_models = {'xgb':[], 'lgb':[], 'cat':[]}

            
        # for i, (X_train_, X_val, y_train_, y_val) in enumerate(splitter.split_data(X_train, y_train, random_state_list=random_state_list)):
        #     n = i % n_splits
        #     m = i // n_splits
                    
        # Get a set of Regressor models
        classifier = Classifier(n_estimators, device, random_state)
        models = classifier.models

        X_train_ = X_train
        y_train_ = y_train
        X_val = X_test
        y_val = y_test

        # Initialize lists to store oof and test predictions for each base model
        names = []
        oof_preds = []
        test_preds = []
        losses = pd.DataFrame(columns=['Model', 'Train Acc', 'Test Acc', 'Test ROC AUC'])

        # Loop over each base model and fit it to the training data, evaluate on validation data, and store predictions
        for name, model in models.items():
            if ('cat' in name) or ("lgb" in name) or ("xgb" in name):
                if 'lgb' == name: #categorical_feature=cat_features
                    model.fit(X_train_, y_train_, eval_set=[(X_val, y_val)],
                              )

                elif 'cat' ==name:
                    model.fit(X_train_, y_train_, eval_set=[(X_val, y_val)],#cat_features=cat_features,
                                early_stopping_rounds=early_stopping_rounds, verbose=verbose)
                elif model.__class__.__name__ == 'LGBMClassifier':
                    model.fit(X_train_, y_train_, eval_set=[(X_val, y_val)],
                                )
                else:
                    model.fit(X_train_, y_train_, eval_set=[(X_val, y_val)], early_stopping_rounds=early_stopping_rounds, verbose=verbose)
                    if name == 'xgb':
                        feature_names = X_train_.columns
                        # Assuming you have a trained model called `model`
                        feature_importance = model.feature_importances_

                        # Sort features and their importances in descending order
                        indices = np.argsort(feature_importance)[::-1]
                        sorted_feature_names = feature_names[indices]
                        sorted_feature_importance = feature_importance[indices]

                        # Plot the top 20 features
                        plt.figure(figsize=(10, 6))
                        sns.barplot(x=sorted_feature_importance[:20], y=sorted_feature_names[:20], palette="viridis")
                        plt.xlabel('Importance', fontsize=14)
                        plt.ylabel('Feature', fontsize=14)
                        plt.title('Top 20 Feature Importance', fontsize=16)
                        plt.grid(axis='x')
                        sns.despine(left=True, bottom=True)
                        plt.show()

            elif name in 'ann':
                model.fit(X_train_, y_train_, validation_data=(X_val, y_val),batch_size=5, epochs=50,verbose=verbose)
            else:
                model.fit(X_train_, y_train_)

            
            if name in 'ann':
                test_pred = np.array(model.predict(X_test))[:, 0]
                y_val_pred = np.array(model.predict(X_val))[:, 0]
            else:
                test_pred = model.predict_proba(X_test)[:, 1]
                y_val_pred = model.predict_proba(X_val)[:, 1]

            score = roc_auc_score(y_val, y_val_pred)
            acc = accuracy_score(y_val, y_val_pred.round())
            train_acc = accuracy_score(y_train_, model.predict(X_train_).round())
            # f1 = f1_score(y_val, y_val_pred.round())
            # precision = precision_score(y_val, y_val_pred.round())
            # recall = recall_score(y_val, y_val_pred.round())
            losses.loc[len(losses)] = [name, train_acc, acc, score]
        #         score = accuracy_score(y_val, acc_cutoff_class(y_val, y_val_pred))

            print(f'{name} ROC AUC score: {score:.2%} | Accuracy {acc:.2%}')
            
            names.append(name)
            oof_preds.append(y_val_pred)
            test_preds.append(test_pred)
            
            if name in trained_models.keys():
                trained_models[f'{name}'].append(deepcopy(model))
        # Use Optuna to find the best ensemble weights

        if best_ensemble:
            model_preds = dict(zip(names, oof_preds))

            study = optuna.create_study(direction='maximize')
            study.optimize(lambda trial: self.optimize_ensemble_weights(trial, oof_preds, y_val), n_trials=100)

            best_weights = study.best_params
            best_weights = [best_weights[f'weight_{i}' ] for i in range(len(oof_preds))]
            best_weights /= np.sum(best_weights)  # Normalize weights to sum to 1

            y_val_pred = np.average(oof_preds, axis=0, weights=best_weights)
            score = roc_auc_score(y_val, y_val_pred)

            losses.loc[len(losses)] = ['Ensemble', score]
            losses.sort_values(by='Test ROC AUC', ascending=False, inplace=True)
            print(f'Ensemble ------------------> ROC AUC score {score:.2%}')
            
            ensemble_score.append(score)
            weights.append(best_weights)

            # Apply the optimized weights to test predictions
            test_predss += np.average(test_preds, axis=0, weights=best_weights) / (n_splits * len(random_state_list))

            weight_to_model = {}

            for i, name in enumerate(names):
                if name in weight_to_model:
                    continue
                weight_to_model[f'weight{i}'] = name

            # Replace weight names with model names
            importance_dict = {weight_to_model.get(name, name): importance for name, importance in optuna.importance.get_param_importances(study).items()}

            # Sort dictionary by values in ascending order
            importance_dict = dict(sorted(importance_dict.items(), key=lambda item: item[1], reverse=True))

            pprint.pprint(importance_dict)

            # Create your own plot using seaborn for better aesthetics
            plt.figure(figsize=(10, 6))
            sns.barplot(x=list(importance_dict.values()), y=list(importance_dict.keys()), palette="viridis")
            plt.xlabel('Importance', fontsize=14)
            plt.ylabel('Model', fontsize=14)
            plt.title('Model Importance', fontsize=16)
            plt.grid(axis='x')
            sns.despine(left=True, bottom=True)
            plt.show()

            gc.collect()

            self.save_log(losses, X_train)
        
        else:
            optweights = OptunaWeights(random_state=random_state)
            y_val_pred = optweights.fit_predict(y_val.values, oof_preds)

            score = roc_auc_score(y_val, y_val_pred)
            acc = accuracy_score(y_val, y_val_pred.round())
            train_acc = accuracy_score(y_train_, model.predict(X_train_).round())
            # f1 = f1_score(y_val, y_val_pred.round())
            # precision = precision_score(y_val, y_val_pred.round())
            # recall = recall_score(y_val, y_val_pred.round())
            losses.loc[len(losses)] = ['Ensemble', train_acc, acc, score]
            losses.sort_values(by='Test Acc', ascending=False, inplace=True)
            #     score = accuracy_score(y_val, acc_cutoff_class(y_val, y_val_pred))
            print(f'Ensemble ------------------>  ROC AUC score {score:.2%}')
            ensemble_score.append(score)
            weights.append(optweights.weights)

            test_predss += optweights.predict(test_preds) / (n_splits * len(random_state_list))

            # Assuming you have a mapping of weight names to model names

            weight_to_model = {}

            for i, name in enumerate(names):
                if name in weight_to_model:
                    continue
                weight_to_model[f'weight{i}'] = name

            # Replace weight names with model names
            importance_dict = {weight_to_model.get(name, name): importance for name, importance in optuna.importance.get_param_importances(optweights.study).items()}

            # Sort dictionary by values in ascending order
            importance_dict = dict(sorted(importance_dict.items(), key=lambda item: item[1], reverse=True))

            pprint.pprint(importance_dict)

            # Create your own plot using seaborn for better aesthetics
            plt.figure(figsize=(10, 6))
            sns.barplot(x=list(importance_dict.values()), y=list(importance_dict.keys()), palette="viridis")
            plt.xlabel('Importance', fontsize=14)
            plt.ylabel('Model', fontsize=14)
            plt.title('Model Importance', fontsize=16)
            plt.grid(axis='x')
            sns.despine(left=True, bottom=True)
            plt.show()

            gc.collect()

            self.save_log(losses, X_train)

        return test_predss, losses
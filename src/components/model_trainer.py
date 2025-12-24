import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV, StratifiedKFold
import os
import sys
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array,test_array):
        try:
            logging.info('Splitting training and test input data')
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            best_models = {
                                'rf': RandomForestClassifier(
                                    n_jobs=1,
                                    random_state=42,
                                    oob_score=False,
                                    bootstrap=True
                                ),
                                'xgb': XGBClassifier(objective="binary:logistic",
                                                    eval_metric="logloss",
                                                    n_jobs=1,
                                                    tree_method="hist",
                                                    random_state=42),
                                'gb': GradientBoostingClassifier(random_state=42),
                                'knn': KNeighborsClassifier(),

                            }
            param_grid = {
                'gb': {
                    'n_estimators': range(50, 301, 20),
                    'learning_rate': np.linspace(0.01, 1, 15),
                    'max_depth': range(5, 10),
                    'min_samples_leaf': np.linspace(0.01, 0.95, 15),
                    'subsample': [0.8, 1.0],
                    'max_features': ['sqrt', 'log2', None]
                },
                'knn': {
                    'n_neighbors': list(range(1, 21)),
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan', 'minkowski']
                },
                'xgb': {
                    'n_estimators': range(50, 500, 20),          # number of boosting rounds
                    'max_depth': range(2, 10),               # tree depth
                    'learning_rate': np.linspace(0.01, 0.3, 20),       # step size shrinkage
                    'subsample': [0.8, 1],            # row sampling
                    'colsample_bytree': np.linspace(0, 0.9, 15),     # feature sampling
                    'gamma': np.linspace(0, 5, 30),                    # min loss reduction
                    'reg_alpha': np.linspace(0, 1, 15),                # L1 regularization
                    'reg_lambda': np.linspace(0, 1, 15)               # L2 regularization
                },
                'rf': {
                    'n_estimators': [200, 300, 500],   # stability > brute force
                    'max_depth': [None, 5, 8, 12, 16], # MOST important parameter
                    'min_samples_split': [2, 5, 10, 20],
                    'min_samples_leaf': [1, 2, 5, 10],
                    'max_features': ['sqrt', 'log2', 0.3, 0.5, 0.7],
                }

            }
            
            logging.info("Training models")

            models_report:dict=evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=best_models, param_grid=param_grid)
            
            best_model_name = max(models_report, key=lambda k: models_report[k][1])

            best_model = models_report[best_model_name][0]

            logging.info("Best model Found:" + best_model_name)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            return models_report[best_model_name][1]
        
        except Exception as e:
            raise CustomException(e, sys)
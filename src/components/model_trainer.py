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
                                                    predictor="cpu_predictor",
                                                    random_state=42),
                                'gb': GradientBoostingClassifier(random_state=42),
                                'knn': KNeighborsClassifier(),
                                'lr1': LogisticRegression(random_state=42),
                                'lr2': LogisticRegression(random_state=42)
                            }
            param_grid = {
                'lr1': {
                    'solver': ['lbfgs', 'sag'],
                    'penalty': ['l2'],
                    'C': np.logspace(-4, 2, 20)
                },
                'lr2': {
                    'solver': ['liblinear'],
                    'penalty': ['l1', 'l2'],
                    'C': np.logspace(-4, 2, 20)
                },
                'knn': {
                    "n_neighbors": list(range(10, 16)),
                    "weights": ["uniform"],
                    "metric": ["minkowski"],
                },
                'gb': {
                    'learning_rate': [0.01, 0.03, 0.05, 0.07],
                    'n_estimators': [300, 400, 500],
                    'min_samples_leaf': [1, 2, 5],
                    'max_depth': [3, 4],
                    'subsample': [0.8, 0.9, 1.0]
                },
                'xgb': {
                    "n_estimators": [350, 450, 500],
                    "learning_rate": [0.01, 0.05, 0.1, 0.2],
                    "max_depth": [6, 7, 8],
                    "gamma": [2.0, 3.0],
                    "subsample": [0.8],
                    "colsample_bytree": [0.6, 0.8],
                    "reg_alpha": [0.1, 0.3, 0.5, 0.7],
                    "reg_lambda": [0.1, 0.3, 0.5, 0.7]
                },
                'rf': {
                    "n_estimators": [300, 400, 500],
                    "max_depth": [5, 8],
                    "min_samples_split": [5, 10, 20],
                    "min_samples_leaf": [1, 2],
                    "max_features": [0.5, 0.7, "sqrt", "log2"],
                }

            }
            
            logging.info("Training models")

            models_report:dict=evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=best_models, param_grid=param_grid)
            
            best_model_name = max(models_report, key=lambda k: models_report[k][1])

            best_model = models_report[best_model_name][0]

            logging.info("Best model Found")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            return models_report[best_model_name][1]
        
        except Exception as e:
            raise CustomException(e, sys)
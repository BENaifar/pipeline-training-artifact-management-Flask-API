import os 
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import dill
from sklearn.metrics import f1_score

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(X_train, y_train, X_test, y_test, models, param_grid):
    try:
        results = {}

        for model in models:
            search_cv = RandomizedSearchCV(models[model], param_distributions=param_grid[model], n_iter=50, scoring='f1', cv=5, random_state=42, n_jobs=-1)

            search_cv.fit(X_train, y_train)

            y_pred = search_cv.best_estimator_.predict(X_test)

            score = f1_score(y_test, y_pred)

            results[model] = [search_cv.best_estimator_, score]
        return results

    except Exception as e:
        raise CustomException(e, sys)


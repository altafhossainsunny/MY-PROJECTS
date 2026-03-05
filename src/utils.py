import os 
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
import pickle
import dill
import joblib
from sklearn.metrics import (f1_score,
                             precision_score,
                             recall_score,
                             accuracy_score,
                             classification_report)

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            joblib.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(X, y, X_test, y_test, models):
    try:
        report = {}
        for i in range(len(models)):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            model.fit(X, y)

            y_test_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_test_pred)

            report[model_name] = accuracy

        return report

    except Exception as e:
        raise CustomException(e, sys)
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier,
                              GradientBoostingClassifier,
                              AdaBoostClassifier)
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import (accuracy_score,
                             f1_score)
import sys
import os
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from src.utils import evaluate_model, save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array, preprocessor_path):
        try:
            logging.info("Splitting training and testing input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Logistic Regression":  LogisticRegression(class_weight='balanced'),
                "SVM":                  SVC(class_weight='balanced'),
                "K-Nearest Neighbors":  KNeighborsClassifier(),
                "Decision Tree":        DecisionTreeClassifier(class_weight='balanced'),
                "Random Forest":        RandomForestClassifier(class_weight='balanced'),
                "Gradient Boosting":    GradientBoostingClassifier(),
                "AdaBoost":             AdaBoostClassifier(),
                "XGBoost":              XGBClassifier(scale_pos_weight=1),
                "LightGBM":             LGBMClassifier(verbose=-1),
                "CatBoost":             CatBoostClassifier(verbose=0)
            }

            model_report = evaluate_model(X=X_train, y=y_train, X_test=X_test, y_test=y_test, models=models)

            # Log all model scores for transparency
            logging.info("=== Model Evaluation Report ===")
            for name, scores in model_report.items():
                logging.info(f"  {name}: Accuracy={scores['accuracy']:.4f}, F1={scores['f1']:.4f}")
            print("\n=== Model Evaluation Report ===")
            for name, scores in model_report.items():
                print(f"  {name:30s} Accuracy={scores['accuracy']:.4f}  F1={scores['f1']:.4f}")

            # Select best model by macro F1 (handles class imbalance better than accuracy)
            best_model_name = max(model_report, key=lambda k: model_report[k]['f1'])
            best_model = models[best_model_name]

            accuracy = model_report[best_model_name]['accuracy']
            f1       = model_report[best_model_name]['f1']

            print(f"\nBest Model: {best_model_name} | Accuracy={accuracy:.4f} | F1={f1:.4f}")

            if accuracy > 0.6 and f1 > 0.3:
                logging.info(f"Best model selected: {best_model_name} | Accuracy={accuracy:.4f} | F1={f1:.4f}")
            else:
                raise CustomException("No suitable model found with required performance metrics.", sys)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            return accuracy

        except CustomException as ce:
            logging.error(f"CustomException occurred: {str(ce)}")
            raise ce

        except Exception as e:
            logging.error(f"Error occurred during model training: {str(e)}")
            raise CustomException(e, sys)
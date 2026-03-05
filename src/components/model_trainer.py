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
from catboost import CatBoostClassifier
from sklearn.metrics import (accuracy_score,
                             classification_report,
                             confusion_matrix,
                             roc_auc_score,
                             f1_score,
                             precision_score)
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
                "Logistic Regression": LogisticRegression(),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "AdaBoost": AdaBoostClassifier(),
                "K-Nearest Neighbors": KNeighborsClassifier(),
                "XGBoost": XGBClassifier(),
                "CatBoost": CatBoostClassifier(verbose=0)
            }

            model_report = evaluate_model(X=X_train, y=y_train, X_test=X_test, y_test=y_test, models=models)

            # Fixed: list(model_report.values()) is a list, use np.argmax directly
            best_model_name = list(model_report.keys())[
                np.argmax(list(model_report.values()))
            ]
            best_model = models[best_model_name]

            predicted = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, predicted)
            f1 = f1_score(y_test, predicted, average='weighted')

            if accuracy > 0.6 and f1 > 0.3:
                logging.info(f"Best model found: {best_model_name} with Accuracy: {accuracy:.4f} and F1 Score: {f1:.4f}")
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
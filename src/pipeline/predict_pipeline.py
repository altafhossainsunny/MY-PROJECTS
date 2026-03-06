import os
import sys
import pandas as pd
import joblib
from src.exception import CustomException
from src.logger import logging

class PredictPipeline:
    def __init__(self):
        self.model_path        = os.path.join('artifacts', 'model.pkl')
        self.preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')

    def predict(self, features: pd.DataFrame):
        try:
            model        = joblib.load(self.model_path)
            preprocessor = joblib.load(self.preprocessor_path)

            data_scaled = preprocessor.transform(features)
            prediction  = model.predict(data_scaled)

            label_map = {0: 'Excellent', 1: 'Fail', 2: 'Good', 3: 'Outstanding'}
            return label_map.get(int(prediction[0]), str(prediction[0]))

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self, gender, race_ethnicity, parental_level_of_education, lunch, test_preparation_course):
        self.gender                      = gender
        self.race_ethnicity              = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch                       = lunch
        self.test_preparation_course     = test_preparation_course

    def get_data_as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([{
            'gender':                      self.gender,
            'race_ethnicity':              self.race_ethnicity,
            'parental_level_of_education': self.parental_level_of_education,
            'lunch':                       self.lunch,
            'test_preparation_course':     self.test_preparation_course
        }])


# ── Predictive Maintenance ────────────────────────────────────────────────────

class MaintenancePredictPipeline:
    def __init__(self):
        self.model_path = os.path.join('artifacts', 'maintenance_model.pkl')

    def predict(self, features: pd.DataFrame):
        try:
            model = joblib.load(self.model_path)
            prediction  = model.predict(features)
            proba       = model.predict_proba(features)          # shape (1, 2)
            failure_pct = round(float(proba[0][1]) * 100, 1)    # probability of class 1
            label = 'FAILURE PREDICTED' if int(prediction[0]) == 1 else 'NORMAL OPERATION'
            return label, failure_pct
        except Exception as e:
            raise CustomException(e, sys)


class MaintenanceData:
    def __init__(self, footfall, tempMode, AQ, USS, CS, VOC, RP, IP, Temperature):
        self.footfall    = footfall
        self.tempMode    = tempMode
        self.AQ          = AQ
        self.USS         = USS
        self.CS          = CS
        self.VOC         = VOC
        self.RP          = RP
        self.IP          = IP
        self.Temperature = Temperature

    def get_data_as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([{
            'footfall':    float(self.footfall),
            'tempMode':    float(self.tempMode),
            'AQ':          float(self.AQ),
            'USS':         float(self.USS),
            'CS':          float(self.CS),
            'VOC':         float(self.VOC),
            'RP':          float(self.RP),
            'IP':          float(self.IP),
            'Temperature': float(self.Temperature),
        }])

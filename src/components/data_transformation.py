import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.dataransformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:

            # Multi-category columns → One-Hot Encoding
            ohe_features = ['parental_level_of_education', 'race_ethnicity']

            # Binary / low-cardinality columns → Ordinal Encoding
            ordinal_features = ['gender', 'lunch', 'test_preparation_course']

            preprocessor = ColumnTransformer(transformers=[
                ('ohe',     OneHotEncoder(handle_unknown='ignore'), ohe_features),
                ('ordinal', OrdinalEncoder(),                       ordinal_features)
            ])
            logging.info("Data encoding completed successfully")
            return preprocessor
        

        except Exception as e:
            logging.error(f"Error occurred while creating data transformer object: {str(e)}")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            preprocessor_obj = self.get_data_transformer_object()

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")

            input_feature_train_df = train_df[['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']]
            train_df['total_score'] = train_df['math_score'] + train_df['reading_score'] + train_df['writing_score']
            target_feature_train_df = train_df['total_score'].apply(
                lambda x: 'fail' if x < 150 else 'good' if x < 241 else 'excellent' if x < 270 else 'outstanding' if x < 301 else 'invalid'
            )

            input_feature_test_df = test_df[['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']]
            test_df['total_score'] = test_df['math_score'] + test_df['reading_score'] + test_df['writing_score']
            target_feature_test_df = test_df['total_score'].apply(
                lambda x: 'fail' if x < 150 else 'good' if x < 241 else 'excellent' if x < 270 else 'outstanding' if x < 301 else 'invalid'
            )
            logging.info("Applying preprocessing object on training and testing data")

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr  = preprocessor_obj.transform(input_feature_test_df)

            label_encoder = LabelEncoder()
            target_train_encoded = label_encoder.fit_transform(np.array(target_feature_train_df))
            target_test_encoded  = label_encoder.transform(np.array(target_feature_test_df))

            logging.info(f"Label classes: {label_encoder.classes_}")

            train_arr = np.c_[input_feature_train_arr, target_train_encoded]
            test_arr  = np.c_[input_feature_test_arr,  target_test_encoded]
            logging.info("Saved preprocessing object.")

            save_object(
                file_path=self.dataransformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            return (
                train_arr,
                test_arr,
                self.dataransformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestion
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_data, test_data)

import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.components.transformers import ColumnAdder, TitleCreator, ColumnDropper

from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Function responsible for creating a data transformation pipeline
        """
        try:
            embarked_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoded', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))

            ])

            sex_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OrdinalEncoder())
            ])

            age_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ])

            numeric_minmax_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', MinMaxScaler())
            ])

            col_trans = ColumnTransformer(
                transformers = [
                    ('Age', age_pipeline, ['Age']),
                    ('Embarked', embarked_pipeline, ['Embarked']),
                    ('Sex', sex_pipeline, ['Sex']),
                    ('minmax', numeric_minmax_pipeline, ['Pclass', 'Ncompanions', 'Fare'])
                ],
                remainder='passthrough',
                verbose_feature_names_out=False
            )

            col_trans.set_output(transform='pandas')

            preprocess_pipeline = Pipeline([
                ('Ncompanions', ColumnAdder(cols=['Parch', 'SibSp'], column_name='Ncompanions')),
                ('Title', TitleCreator()),
                ('transformed', col_trans),
                ('dropper', ColumnDropper(cols=['PassengerId', 'Name', 'Ticket', 'Cabin', 'Title']))
            ])

            preprocess_pipeline.set_output(transform='pandas')

            logging.info("Transormations Pipeline created")

            return preprocess_pipeline


        except Exception as e:
            raise CustomException(e, sys)

    def init_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "Survived"

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training dataframe and testing dataframe.")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                np.array(input_feature_train_arr),
                np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                np.array(input_feature_test_arr),
                np.array(target_feature_test_df)
            ]

            logging.info('Saved preprocessing object.')

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
            )

        except Exception as e:
            raise CustomException(e, sys)
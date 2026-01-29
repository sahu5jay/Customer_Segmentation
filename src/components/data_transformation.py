import pandas as pd
import numpy as np
import os
import sys

from src.exception import CustomException
from src.logger import logging

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from src.utils import save_object
from dataclasses import dataclass


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file = os.path.join('artifacts', 'preprocessor.joblib')


class DataTransformation:
    def __init__(self):
        self.data_transformation = DataTransformationConfig()

    def get_preprocessor_obj(self):

        try:
            logging.info("Data transformation initiated")

            df = pd.read_csv(os.path.join('artifacts', 'raw.csv'), sep="\t")
            logging.info("Raw data converted to dataframe from artifacts")

            # ---------------- Feature Engineering (from IPYNB) ----------------
            df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"])
            df["Customer_Tenure"] = (pd.Timestamp.today() - df["Dt_Customer"]).dt.days

            df["Total_Spending"] = (
                df["MntWines"] +
                df["MntFruits"] +
                df["MntMeatProducts"] +
                df["MntFishProducts"] +
                df["MntSweetProducts"] +
                df["MntGoldProds"]
            )

            drop_cols = ["ID", "Dt_Customer"]
            df = df.drop(columns=drop_cols, errors="ignore")

            num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
            cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('scaler', StandardScaler())
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', num_pipeline, num_cols),
                    ('cat_pipeline', cat_pipeline, cat_cols)
                ]
            )

            logging.info("Returning the preprocessor pipeline")
            return preprocessor

        except Exception as e:
            logging.info("Exception raised inside get_preprocessor_obj function")
            raise CustomException(e, sys)

    def initate_data_transformation(self, train_path, test_path):

        try:
            logging.info("Fetching the path of the train and test data")

            train_df = pd.read_csv(train_path, sep="\t")
            test_df = pd.read_csv(test_path, sep="\t")

            logging.info("Train and test data converted to dataframe")

            preprocessor_obj = self.get_preprocessor_obj()
            logging.info("Preprocessor object collected")

            target_col = 'Cluster'   # created later by clustering
            drop_col = [target_col]

            logging.info("Separating input features and target feature from train dataset")
            input_feature_train_df = train_df.drop(columns=drop_col, axis=1, errors="ignore")
            target_feature_train_df = train_df[target_col] if target_col in train_df else None

            logging.info("Separating input features and target feature from test dataset")
            input_feature_test_df = test_df.drop(columns=drop_col, axis=1, errors="ignore")
            target_feature_test_df = test_df[target_col] if target_col in test_df else None

            logging.info("Fit-transform on the training dataset")
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)

            logging.info("Transform on the test dataset")
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            if target_feature_train_df is not None:
                logging.info("Concatenating training features and target")
                train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            else:
                train_arr = input_feature_train_arr

            if target_feature_test_df is not None:
                logging.info("Concatenating test features and target")
                test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            else:
                test_arr = input_feature_test_arr

            logging.info("Saving the preprocessor object")
            save_object(
                file_path=self.data_transformation.preprocessor_obj_file,
                obj=preprocessor_obj
            )

            logging.info("Returning train array, test array, and preprocessor file path")
            return (
                train_arr,
                test_arr,
                self.data_transformation.preprocessor_obj_file
            )

        except Exception as e:
            logging.info("Exception occurred inside initiate_data_transformation function")
            raise CustomException(e, sys)

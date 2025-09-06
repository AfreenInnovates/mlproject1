# src/components/data_transformation.py
import os
import sys
import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifact", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def get_data_transformer_object(self) -> ColumnTransformer:
        """
        Create a ColumnTransformer that:
          - imputes + scales numeric features
          - imputes + one-hot-encodes (then scales) categorical features
        """
        try:
            target_col = "GradeClass"
            id_cols = ["StudentID"]

            numeric_features = [
                "Age",
                "StudyTimeWeekly",
                "Absences",
                "GPA",
            ]

            categorical_features = [
                "Gender",
                "Ethnicity",
                "ParentalEducation",
                "Tutoring",
                "ParentalSupport",
                "Extracurricular",
                "Sports",
                "Music",
                "Volunteering",
            ]

            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
                # Scale after one-hot; with_mean=False keeps sparse matrices valid
                ("scaler", StandardScaler(with_mean=False)),
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", num_pipeline, numeric_features),
                    ("cat", cat_pipeline, categorical_features),
                ],
                remainder="drop",  # drop id/target/anything not listed
            )

            logging.info("Created data transformation preprocessor")
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path: str, test_path: str):
        """
        Reads train/test CSVs, fits the preprocessor on train, transforms both,
        saves the preprocessor, and returns numpy arrays ready for modeling.
        """
        try:
            logging.info("Reading train and test data")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            target_col = "GradeClass"
            id_cols = ["StudentID"]

            X_train = train_df.drop(columns=[target_col] + id_cols, errors="ignore")
            y_train = train_df[target_col]

            X_test = test_df.drop(columns=[target_col] + id_cols, errors="ignore")
            y_test = test_df[target_col]

            logging.info("Building preprocessor")
            preprocessor = self.get_data_transformer_object()

            logging.info("Fitting preprocessor on training data")
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            os.makedirs(os.path.dirname(self.config.preprocessor_obj_file_path), exist_ok=True)
            with open(self.config.preprocessor_obj_file_path, "wb") as f:
                pickle.dump(preprocessor, f)
            logging.info(f"Saved preprocessor to {self.config.preprocessor_obj_file_path}")


            train_arr = np.c_[X_train_transformed.toarray() if hasattr(X_train_transformed, "toarray") else X_train_transformed,
                              np.array(y_train)]
            test_arr = np.c_[X_test_transformed.toarray() if hasattr(X_test_transformed, "toarray") else X_test_transformed,
                             np.array(y_test)]

            return (train_arr, test_arr, self.config.preprocessor_obj_file_path)

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    transformer = DataTransformation()
    train_p = os.path.join("artifact", "train.csv")
    test_p = os.path.join("artifact", "test.csv")
    transformer.initiate_data_transformation(train_p, test_p)

import os
import sys

import pandas as pd

from src.exception import CustomException
from src.logger import logging


class DataValidation:

    def __init__(self):
        self.required_columns = [
            "UDI",
            "Product ID",
            "Type",
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]",
            "Machine failure"
        ]

    def validate_file_exists(self):
        try:
            logging.info("Checking if train and test files exist")

            train_file_status = os.path.exists("artifacts/train.csv")
            test_file_status = os.path.exists("artifacts/test.csv")

            if not train_file_status:
                raise FileNotFoundError("train.csv not found")

            if not test_file_status:
                raise FileNotFoundError("test.csv not found")

            logging.info("Train and test files validated successfully")

        except Exception as e:
            raise CustomException(e, sys)

    def validate_dataset_not_empty(self):
        try:
            logging.info("Checking if datasets are empty")

            train_df = pd.read_csv("artifacts/train.csv")
            test_df = pd.read_csv("artifacts/test.csv")

            if train_df.empty:
                raise ValueError("Train dataset is empty")

            if test_df.empty:
                raise ValueError("Test dataset is empty")

            logging.info("Datasets are not empty")

        except Exception as e:
            raise CustomException(e, sys)

    def validate_required_columns(self):
        try:
            logging.info("Validating required columns")

            train_df = pd.read_csv("artifacts/train.csv")

            missing_columns = [
                col for col in self.required_columns
                if col not in train_df.columns
            ]

            if missing_columns:
                raise ValueError(
                    f"Missing columns: {missing_columns}"
                )

            logging.info("All required columns are present")

        except Exception as e:
            raise CustomException(e, sys)
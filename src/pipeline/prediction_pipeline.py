import os
import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


class PredictPipeline:

    def predict(self, features):

        try:

            model_path = os.path.join(
                "artifacts",
                "model.pkl"
            )

            preprocessor_path = os.path.join(
                "artifacts",
                "preprocessor.pkl"
            )

            model = load_object(model_path)
            preprocessor = load_object(
                preprocessor_path
            )

            scaled_data = preprocessor.transform(
                features
            )

            prediction = model.predict(
                scaled_data
            )[0]

            probability = model.predict_proba(
                scaled_data
            )[0][1]

            return prediction, probability

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:

    def __init__(
        self,
        Type,
        Air_temperature,
        Process_temperature,
        Rotational_speed,
        Torque,
        Tool_wear
    ):

        self.Type = Type

        self.Air_temperature = Air_temperature

        self.Process_temperature = (
            Process_temperature
        )

        self.Rotational_speed = (
            Rotational_speed
        )

        self.Torque = Torque

        self.Tool_wear = Tool_wear

    def get_data_as_dataframe(self):

        try:

            data = {
                "Type": [self.Type],
                "Air temperature [K]":
                    [self.Air_temperature],
                "Process temperature [K]":
                    [self.Process_temperature],
                "Rotational speed [rpm]":
                    [self.Rotational_speed],
                "Torque [Nm]":
                    [self.Torque],
                "Tool wear [min]":
                    [self.Tool_wear]
            }

            return pd.DataFrame(data)

        except Exception as e:
            raise CustomException(e, sys)
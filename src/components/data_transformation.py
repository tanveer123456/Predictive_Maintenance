import os
import sys
import pandas as pd
import numpy as np  
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline   
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.logger import logging
from src.exception import CustomException
from src.components.data_transformation_config import DataTransformationConfig
from src.utils import save_object
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    def get_data_transformer_object(self):
        try:
            categorical_cols=["Type"]
            numerical_cols=[ "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]"]
            num_pipeline=Pipeline(
                steps=[
                    ("scaler",StandardScaler())
                ]
            )
            cat_pipeline=Pipeline(
                steps=[
                    ("one_hot_encoder",OneHotEncoder(handle_unknown="ignore")),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )
            preprocessor=ColumnTransformer(
                transformers=[
                    ("num_pipeline",num_pipeline,numerical_cols),
                    ("cat_pipeline",cat_pipeline,categorical_cols)
                ]
            )
            logging.info("Preprocessor object created successfully")
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            logging.info("Reading train and test data")

            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessor object")
            preprocessor_obj=self.get_data_transformer_object()
            target_column_name="Machine failure"
            drop_columns=["UDI","Product ID"]
            input_feature_train_df=train_df.drop(columns=drop_columns+[target_column_name])
            target_feature_train_df=train_df[target_column_name]
            input_feature_test_df=test_df.drop(columns=drop_columns+[target_column_name])
            target_feature_test_df=test_df[target_column_name]
            logging.info("Applying preprocessor object on training and testing data")
            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)                
            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
            logging.info("Saved preprocessor object")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )                   
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)    

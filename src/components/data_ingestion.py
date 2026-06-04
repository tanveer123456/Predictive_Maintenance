import os 
import sys

import pandas as pd 
from sklearn.model_selection import train_test_split
from src.components.data_ingestion_config import DataIngestionConfig
from src.logger import logging
from src.exception import CustomException
class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            logging.info("Entered the data ingestion method")
            df=pd.read_csv("data/data.csv")
            logging.info("Dataset read successfully into dataframe")
            os.makedirs("artifacts",exist_ok=True)
            df.to_csv(
                self.ingestion_config.raw_data_path,
                index=False,
                header=True)
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            logging.info("Train test split completed")
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("Data ingestion completed successfully")
        except Exception as e:
            raise CustomException(e,sys)

        

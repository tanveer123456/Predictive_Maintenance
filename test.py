from src.components.data_ingestion import DataIngestion
import pandas as pd


obj = DataIngestion()

obj.initiate_data_ingestion()

# raw_df=pd.read_csv('artifacts/raw.csv')
# print(raw_df.head())
# train_df=pd.read_csv('artifacts/train.csv')
# print(train_df.shape)
# test_df=pd.read_csv('artifacts/test.csv')
# print(test_df.shape)
from src.components.data_validation import DataValidation

obj = DataValidation()
obj.validate_file_exists()
obj.validate_dataset_not_empty()
obj.validate_required_columns()

import pandas as pd

df = pd.read_csv("data/data.csv")
print(df["Machine failure"].value_counts())
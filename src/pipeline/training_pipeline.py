from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation   
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logging
from src.exception import CustomException
import sys
class TrainingPipeline:
    def __init__(self):
        pass
    def start_training_pipeline(self):
        try:
           logging.info("Starting the training pipeline")

           data_ingestion=DataIngestion()
           data_ingestion.initiate_data_ingestion()
           logging.info("Data ingestion completed successfully")
           data_validation=DataValidation()
           data_validation.validate_file_exists()
           data_validation.validate_dataset_not_empty()
           data_validation.validate_required_columns()
           logging.info("Data validation completed successfully")
           data_transformation = DataTransformation()

           train_arr, test_arr, _ = (
               data_transformation.initiate_data_transformation(
                   train_path="artifacts/train.csv",
                   test_path="artifacts/test.csv"
               )
           )
           logging.info("Data transformation completed successfully")
           model_trainer=ModelTrainer()
           model_score=model_trainer.initiate_model_trainer(train_array=train_arr,test_array=test_arr)
           logging.info(f"Model training completed successfully with best model score: {model_score}")
           return model_score
        except Exception as e:
            raise CustomException(e,sys)
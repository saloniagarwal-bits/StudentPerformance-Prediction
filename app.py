from src.performanceprediction.logger import logging
from src.performanceprediction.exception import CustomException
from src.performanceprediction.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.performanceprediction.components.data_transformation import DataTransformationConfig, DataTransformation
import sys

if __name__=="__main__":
    logging.info("Execution started")

    try:
        data_ingestion=DataIngestion()
        train_data_path , test_data_path= data_ingestion.initiate_data_ingestion()

        data_transformation_config = DataIngestionConfig()
        data_transformation = DataTransformation()
        data_transformation.initiate_data_transformation(train_data_path, test_data_path)


    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e,sys)
from src.performanceprediction.logger import logging
from src.performanceprediction.exception import CustomException
from src.performanceprediction.components.data_ingestion import DataIngestion
import sys

if __name__=="__main__":
    logging.info("Execution started")

    try:
        data_ingestion=DataIngestion()
        data_ingestion.initiate_data_ingestion()

    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e,sys)
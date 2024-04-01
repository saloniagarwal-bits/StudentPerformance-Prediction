from src.performanceprediction.logger import logging
from src.performanceprediction.exception import CustomException
import sys

if __name__=="__main__":
    logging.info("Execution started")

    try:
        a=1/0
    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e,sys)
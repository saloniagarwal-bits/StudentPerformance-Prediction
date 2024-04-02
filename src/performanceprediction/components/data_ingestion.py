import os
import sys
from src.performanceprediction.exception import CustomException
from src.performanceprediction.logger import logging
import pandas as pd
from src.performanceprediction.utils import read_sql_data
from sklearn.model_selection import train_test_split

from dataclasses import dataclass
#data class is used to define some parameters

@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            #read data from mysql
            # df=read_sql_data()
            df=pd.read_csv(self.ingestion_config.raw_data_path)
            logging.info("Read csv data started")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            # df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Read csv data done")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as ex:
            raise CustomException(ex,sys)
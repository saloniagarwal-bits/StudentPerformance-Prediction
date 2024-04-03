import os
import sys
from src.performanceprediction.exception import CustomException
from src.performanceprediction.logger import logging
import pandas as pd
from dotenv import load_dotenv
import pymysql
import pickle
import numpy

load_dotenv()

host=os.getenv("host")
user=os.getenv("user")
password=os.getenv("password")
db=os.getenv("db")

#This file has common functionality

def read_sql_data():
    logging.info('Reading SQL database started')
    try:
        mydb=pymysql.connect(
            host=host,
            user=user,
            password=password,
            db=db
        )
        logging.info('Connection done', mydb)
        df=pd.read_sql_query('Select * from student', mydb)
        print(df.head())
        return df

    except Exception as e:
        raise CustomException(e,sys)
    
'''
This will create a pickle file for objects like preprocessing, models etc.
'''
def save_object(file_path, obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as ex:
        raise CustomException(ex,sys)

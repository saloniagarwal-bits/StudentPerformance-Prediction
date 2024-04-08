import os
import sys
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from src.performanceprediction.exception import CustomException
from src.performanceprediction.logger import logging
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
import pandas as pd
from dotenv import load_dotenv
import pymysql
import pickle

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
    
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for i in range(len(models)):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]

            grid_cv = GridSearchCV(model, para, cv=3)
            grid_cv.fit(X_train, y_train)

            model.set_params(**grid_cv.best_params_)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_score = r2_score(y_train, y_train_pred)
            test_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_score

        return report

    except Exception as ex:
        raise CustomException(ex, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)

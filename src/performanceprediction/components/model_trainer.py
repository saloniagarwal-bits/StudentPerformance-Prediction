from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import sys, os
from dataclasses import dataclass
from src.performanceprediction.utils import save_object,evaluate_models

from src.performanceprediction.exception import CustomException
from src.performanceprediction.logger import logging

@dataclass
class ModelTrainerConfig:
    train_model_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_train_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('Split training and test input data')

            X_train = train_array[:,:-1]
            y_train = train_array[:,-1]
            X_test = test_array[:,:-1]
            y_test = test_array[:,-1]

            models= {
                "Linear Regression" : LinearRegression(),
                "Gradient Boosting" : GradientBoostingRegressor(),
                "Decision Tree" : DecisionTreeRegressor(),
                "Random Forest" : RandomForestRegressor(),
                "XGBRegressor" : XGBRegressor(),
                "CatBoosting Regressor" : CatBoostRegressor(allow_writing_files=False,silent=True),
                "AdaBoost Regressor" : AdaBoostRegressor()
            }

            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }

            model_report = evaluate_models(X_train, y_train, X_test, y_test, models, params)

            print(model_report)

            #Get best model score
            best_model_score = max(sorted(model_report.values()))
            best_model_name= list(models.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model_object = models[best_model_name]

            print(best_model_name)

            if best_model_score<0.6:
                raise CustomException('No best model found')
            logging.info('Best model found on train and test dataset')

            save_object(
                file_path=self.model_train_config.train_model_path,
                obj=best_model_object
            )

            predicted = best_model_object.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            return r2_square



        except Exception as ex:
            raise CustomException(ex,sys)
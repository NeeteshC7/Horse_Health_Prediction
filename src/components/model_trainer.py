import os
import sys
from dataclasses import dataclass

# Sklearn
from sklearn.preprocessing import StandardScaler, QuantileTransformer, OrdinalEncoder
from sklearn.compose import  make_column_selector
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import FunctionTransformer


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import warnings

from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import accuracy_score

from src.utils import save_object_to_file,evaluate_models


from dataclasses import dataclass
@dataclass
class ModelTrainingConfig:
    model_filepath = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.training_config = ModelTrainingConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array.iloc[:, :-1],
                train_array.iloc[:, -1] ,
                test_array.iloc[:, :-1],
                test_array.iloc[:, -1]
            )
            models = {
                "Random Forest": RandomForestClassifier()
            }

            params = {
                "Random Forest": {
                    'n_estimators': [50, 100, 150, 170, 200, 220],
                    'max_depth': [None, 5, 10, 15, 20],
                    'min_samples_split': [ 2, 3, 5, 7],
                    'min_samples_leaf': [ 2, 3, 4],
                    'bootstrap': [True, False]
                }
            }

            model_report = evaluate_models(
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                models=models, param=params
            )

            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]

            if model_report[best_model_name] < 0.6:
                raise CustomException("No best model found")

            logging.info(f"Using Random forest model on both training and testing dataset")

            save_object_to_file(
                file_path=self.training_config.model_filepath,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            logging.info(f"First row of Test Dataset : {X_test.iloc[0].values.tolist() }")

            accuracy_scores = accuracy_score(y_test, predicted)

            logging.info(f"Accuracy Score of Random Forest is {accuracy_scores}" )
            return accuracy_scores

        
        
        
        
        
        
        
        
        
        except Exception as e:
            raise CustomException(e,sys)

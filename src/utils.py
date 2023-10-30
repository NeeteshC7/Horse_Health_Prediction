import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score

from src.exception import CustomException

def save_object_to_file(file_path, obj):
    try:
        # Get the directory path from the file path
        directory_path = os.path.dirname(file_path)
        
        # Create the directory if it doesn't exist
        os.makedirs(directory_path, exist_ok=True)
        
        # Open the file in binary write mode
        with open(file_path, "wb") as file_object:
            # Serialize and save the object to the file
            pickle.dump(obj, file_object)
    
    except Exception as error:
        raise CustomException(error, sys)
    

def evaluate_models( X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            # Initialize RandomizedSearchCV
            random_search = RandomizedSearchCV(
                model, param_distributions=para, n_iter=25, cv=8, verbose=1, random_state=42, n_jobs=-1
            )

            random_search.fit(X_train,y_train)

            model.set_params(**random_search.best_params_)
            model.fit(X_train,y_train)

            #model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = accuracy_score(y_train, y_train_pred)

            test_model_score = accuracy_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
    

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
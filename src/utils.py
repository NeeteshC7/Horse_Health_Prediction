import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle

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
    
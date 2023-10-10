import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder,StandardScaler, QuantileTransformer

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object_to_file

import pdb

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self, train_df):
        try:

            numerical_columns = train_df.select_dtypes(exclude='object').columns.tolist()
            categorical_columns = train_df.select_dtypes(include='object').columns.tolist()

            columns_to_remove = ['lesion_3', 'outcome'] #remove the target column and others which are later removed

            numerical_columns = [col for col in numerical_columns if col not in columns_to_remove]
            categorical_columns = [col for col in categorical_columns if col not in columns_to_remove]


            # numerical_columns =  ['hospital_number', 'rectal_temp', 'pulse', 'respiratory_rate', 'nasogastric_reflux_ph', 'packed_cell_volume', 'total_protein', 'abdomo_protein', 'lesion_1', 'lesion_2']
            # categorical_columns =  ['surgery', 'age', 'temp_of_extremities', 'peripheral_pulse', 'mucous_membrane', 'capillary_refill_time', 'pain', 'peristalsis', 'abdominal_distention', 'nasogastric_tube', 'nasogastric_reflux', 'rectal_exam_feces', 'abdomen', 'abdomo_appearance', 'surgical_lesion', 'cp_data']

            numerical_pipeline = Pipeline([
                ('imputer', IterativeImputer(max_iter=50, random_state=0)),  
                ('quantile_transform', QuantileTransformer(output_distribution='normal', random_state=42)),
                ('standard_scaler', StandardScaler())
            ])

            categorical_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),  
                ('ordinal_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=10))
            ])

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", numerical_pipeline, numerical_columns),
                    ("cat_pipelines", categorical_pipeline, categorical_columns)
                ],
                remainder='passthrough', 
                verbose_feature_names_out=False  
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        

    def initiate_data_transformation(self,train_path,test_path):

            try:
                train_df=pd.read_csv(train_path)
                test_df=pd.read_csv(test_path)

                logging.info("Reading train and test data")

                logging.info("Obtaining preprocessing object")

                preprocessing_obj=self.get_data_transformer_object(train_df)

                target_column_name='outcome'
                columns_to_drop = ['lesion_3']
                columns_to_drop.append(target_column_name)


                input_feature_train_df=train_df.drop(columns=columns_to_drop  ,axis=1)
                target_feature_train_df=train_df[target_column_name]
                target_feature_train_df = target_feature_train_df.map({'died': 0, 'euthanized': 1, 'lived': 2}.get)

                input_feature_test_df=test_df.drop(columns=columns_to_drop ,axis=1)
                target_feature_test_df=test_df[target_column_name]
                target_feature_test_df = target_feature_test_df.map({'died': 0, 'euthanized': 1, 'lived': 2}.get)

                logging.info(
                    f"Applying preprocessing object on training dataframe and testing dataframe."
                )

                input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
                input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

                train_arr = np.c_[
                    input_feature_train_arr, np.array(target_feature_train_df)
                ]
                test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

                logging.info(f"Saved preprocessing object.")

                save_object_to_file(

                    file_path=self.data_transformation_config.preprocessor_obj_file_path,
                    obj=preprocessing_obj

                )

                return (
                    train_arr,
                    test_arr,
                    self.data_transformation_config.preprocessor_obj_file_path,
                )
            except Exception as e:
                raise CustomException(e,sys)
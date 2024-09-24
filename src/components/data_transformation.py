#Data transformation

import sys 
from dataclasses import dataclass


import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object



@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','processor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    #Function is responsible for data transformation
    def get_data_transformer_obj(self):
        try:
            numerical_columns = ["writing score","reading score"]
            categorical_columns = ["gender", "race/ethnicity", "parental level of education","lunch","test preparation course"]

            num_pipeline =Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),#Handling missing values
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy = "most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler())

                ]
            )
            
            logging.info("Categorical and numerical column encoding and standardising")
            
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline,numerical_columns),
                    ("cat_pipeline",cat_pipeline, categorical_columns)
                ]

            )

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self, train_path,test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data")

            preprocessing_obj = self.get_data_transformer_obj()

            target_column_name = "math score"
            numerical_columns = ["writing score", "reading score"]

            input_feature_train = train_df.drop(columns= [target_column_name],axis =1)
            target_feature_train = train_df[target_column_name]

            input_feature_test = test_df.drop(columns=[target_column_name],axis = 1)
            target_feature_test = test_df[target_column_name]

            logging.info(f"applying preprocessing methods on test and train data sets")

            input_feature_arr_train = preprocessing_obj.fit_transform(input_feature_train)
            input_feature_arr_test = preprocessing_obj.transform(input_feature_test)
            #Difference - transform() is used to apply the learned transformation so used in test, where as fit() learns the parameters from ther training data
            #fit_transform() does both, fit() is more convenient when there are multiple datasets

            train_arr = np.c_[
                input_feature_arr_train, np.array(target_feature_train)
            ]

            test_arr = np.c_[input_feature_arr_test, np.array(target_feature_test)]

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return(train_arr,test_arr, self.data_transformation_config.preprocessor_obj_file_path)


        except Exception as e:
            raise CustomException(e, sys)
import sys
from dataclasses import dataclass
import os

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler,LabelEncoder
  
from src.utils.exception import CustomException
from src.utils.logger import logging

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
    def get_data_transformer_object(self, df:pd.DataFrame,target_column_name:str):
        try:
            features_df=df.drop(columns=[target_column_name],axis=1)
            numerical_colums=df.select_dtypes(include=['int64','float64']).columns.tolist()
            categorical_columns=df.select_dtypes(include=['object','categorical']).columns.tolist()

            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='median')),
                    ("Scaler",StandardScaler())
                ]
            )

            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("Scaler",StandardScaler(with_mean=False))
                ]
            )
            logging.info(f"Categorical colums: {categorical_columns}")
            logging.info(f"numerical columns: {numerical_colums}")

            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_colums),
                    ("cat_pipeline",cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor
        
        except Exception as e:
           raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info('read train and test data completed')
            logging.info('obtaining preprocessing object')


            target_column_name  =''  #handled by frontend
            preprocessing_obj=self.get_data_transformer_object(train_df,target_column_name)

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            if target_feature_train_df.dtype=='object':
                label_encoder=label_encoder()
                target_feature_train_df=label_encoder.fit_transform(target_feature_train_df)
                target_feature_test_df=label_encoder.transform(target_feature_test_df)

            logging.info(f"applying preprocessing object on trainig dataframe and testing dataframe")
            
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df,target_feature_train_df)
            input_feature_test_arr=preprocessing_obj.fit_transform(input_feature_test_df,target_feature_test_df)
 
            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
            
            logging.info(f"Saved preprocessing object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
           raise CustomException(e,sys)

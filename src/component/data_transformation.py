# this file contains the data transformation functions
# eg changing catergorical data to numerical data
# eg handling one hot encoding
# eg handling missing data


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object



'''
This defines a dataclass named DataTransformationConfig using the @dataclass decorator. 
It has one attribute preprocessor_obj_file_path which is initialized with a file path where 
the preprocessor object will be saved. 
os.path.join is used to construct the file path in a platform-independent manner.
'''
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")


'''This defines a class named DataTransformation. Its constructor (__init__ method) initializes an instance of DataTransformationConfig and assigns it to the attribute data_transformation_config. This attribute holds configuration parameters related to data transformation.'''
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data trnasformation
        It's responsible for creating and returning a preprocessor object, which is essential for data transformation.
        '''
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            '''It creates a pipeline (num_pipeline) for numerical columns. This pipeline consists of two steps:
                Imputation (SimpleImputer): Missing values are filled using the median value of the respective column.
                Scaling (StandardScaler): The numerical features are standardized by removing the mean and scaling to unit variance.'''
            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())

                ]
            )

            '''
             it creates a pipeline (cat_pipeline) for categorical columns. This pipeline includes the following steps:
            Imputation (SimpleImputer): Missing values are filled using the most frequent value of the respective column.
            One-Hot Encoding (OneHotEncoder): Categorical features are converted into one-hot encoded binary arrays.
            Scaling (StandardScaler): The categorical features are standardized without centering them (with_mean=False).
            '''
            cat_pipeline=Pipeline(

                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
                ]

            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            '''It creates a ColumnTransformer named preprocessor. This transformer applies different preprocessing pipelines to numerical and categorical columns based on the specified lists (numerical_columns and categorical_columns).'''
            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns)

                ]


            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
    
    '''It's responsible for orchestrating the data transformation process using the preprocessor object.'''
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="math_score"
            numerical_columns = ["writing_score", "reading_score"]

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )


            '''It applies the preprocessor object to transform the input features of both the training 
            and testing datasets using the fit_transform and transform methods, respectively.'''
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)


            '''It combines the transformed input features with the target features as NumPy arrays (train_arr and test_arr).'''
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")


            '''It saves the preprocessor object to a file using the save_object function from the custom module.'''
            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            '''Finally, it returns the transformed training and testing arrays along with the file path where the preprocessor object is saved.'''
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)
        

'''

explaination of code:

import sys: This imports the Python sys module, which provides access to some variables used or maintained by the Python interpreter and to functions that interact strongly with the interpreter. In this script, it's likely used for system-specific functionality or for accessing command-line arguments.

from dataclasses import dataclass: This imports the dataclass decorator from the dataclasses module. Data classes are a feature added in Python 3.7 that provide a way to create classes with automatically generated special methods, such as __init__ and __repr__, based on class variable definitions.

ColumnTransformer: A transformer that applies different transformations to different columns of a dataset.
SimpleImputer: A transformer for imputing missing values in a dataset, using simple strategies like mean, median, most frequent, etc.
Pipeline: A class for chaining multiple transformers and an estimator into one pipeline.
OneHotEncoder: A transformer for converting categorical integer features into one-hot encoded binary arrays.
StandardScaler: A transformer for standardizing features by removing the mean and scaling to unit variance.

import os: This imports the os module, which provides a portable way of using operating system-dependent functionality, such as reading or writing to the file system.
'''



'''

1)input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
preprocessing_obj: This is the preprocessor object obtained from the get_data_transformer_object method. It is a ColumnTransformer that encapsulates the preprocessing steps for both numerical and categorical features.
.fit_transform(input_feature_train_df): This method call performs two essential steps: fitting the transformer to the training data and transforming the training data simultaneously. Here's what happens in detail:
Fitting: The fit_transform method first fits the preprocessing steps defined in preprocessing_obj to the training data. During this step, the transformer learns statistics (like the median for imputation or mean and standard deviation for scaling) from the training data.
Transforming: After fitting, the method transforms the training data. It applies the learned transformations (imputation and scaling) to the input features (input_feature_train_df). As a result, missing values are filled, and numerical features are scaled based on the learned statistics.

2)input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
.transform(input_feature_test_df): This method call applies the learned transformations (imputation and scaling) from the fitted preprocessing_obj to the testing data (input_feature_test_df). Here's what happens:
The transform method only applies transformations to the testing data without re-fitting the transformer. It uses the same statistics learned during the fitting stage on the training data.
This ensures that the testing data undergoes the same preprocessing steps as the training data, maintaining consistency in data processing.

Overall, these lines of code demonstrate the typical workflow of using scikit-learn's transformers:

Fitting: Fit the transformer to the training data to learn statistics and parameters.
Transforming: Apply the learned transformations to both training and testing data using the fit_transform method for training data and the transform method for testing data.


'''
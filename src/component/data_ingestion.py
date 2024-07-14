# this file will contain the code to read the data from different sources

'''

Data ingestion refers to the process of importing, collecting, or loading data into a system or storage environment for
further processing, analysis, or storage. It's a fundamental step in data management pipelines and is often the 
first step in various data-related tasks such as data analysis, machine learning, data warehousing, and business intelligence.
'''



'''
This code defines a data ingestion process along with subsequent steps for data transformation and model training. 
Let's go through it in detail:

Imports:
os, sys: Standard Python modules for system-level operations.
CustomException: An exception class defined in the src.exception module.
logging: A custom logging module imported from src.logger.
pandas as pd: Importing the pandas library to work with data in DataFrame format.
train_test_split: Function from Scikit-learn to split data into training and test sets.
dataclass: Decorator to simplify the creation of classes that primarily store data.
DataTransformation, DataTransformationConfig: Classes for data transformation defined in src.components.data_transformation.
ModelTrainer, ModelTrainerConfig: Classes for model training defined in src.components.model_trainer.

DataIngestionConfig Dataclass:
Defines a dataclass DataIngestionConfig with three attributes:
train_data_path: Path to the training data CSV file.
test_data_path: Path to the test data CSV file.
raw_data_path: Path to store the raw data CSV file.

DataIngestion Class:
Initializes with an instance of DataIngestionConfig.
Defines a method initiate_data_ingestion() for data ingestion:
Logs the entry into the data ingestion method.
Attempts to read a CSV file named 'stud.csv' into a pandas DataFrame.
Logs the successful reading of the dataset.
Creates directories for storing data if they don't exist.
Saves the entire dataset as raw data CSV using the path specified in raw_data_path.
Performs train-test split on the dataset with a test size of 20%.
Saves the training and test sets as separate CSV files using paths specified in train_data_path and test_data_path.
Logs the completion of data ingestion.
Returns the paths of the training and test data CSV files.
Raises a CustomException if any exception occurs during the process.

Main Section:
Instantiates the DataIngestion class as obj.
Calls the initiate_data_ingestion() method to ingest data and retrieves paths to training and test data.
Instantiates DataTransformation and ModelTrainer classes.
Calls initiate_data_transformation() method of DataTransformation to transform data.
Calls initiate_model_trainer() method of ModelTrainer to train the model and prints the result.

Overall:
This code orchestrates the process of ingesting raw data from a CSV file, 
splitting it into training and test sets, transforming the data, and training a model.
 It encapsulates each step into separate classes and methods, making the code modular and 
 easier to maintain. Additionally, it utilizes exception handling and logging for error tracking 
 and debugging.

'''

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass  #used to create class variable

from src.component.data_transformation import DataTransformation
from src.component.data_transformation import DataTransformationConfig
from src.exception import CustomException

from src.component.model_trainer import ModelTrainer
from src.component.model_trainer import ModelTrainerConfig


@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df=pd.read_csv('notebook\data\stud.csv')
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train test split initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Ingestion of the data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))
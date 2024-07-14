'''
These lines import necessary libraries and modules:
os: For interacting with the operating system, used for file operations.
sys: For system-specific parameters and functions, used for exception handling.
numpy: For numerical computing in Python, providing support for arrays and mathematical functions.
pandas: For data manipulation and analysis, providing DataFrame structures.
dill: For serializing and deserializing Python objects, an alternative to pickle.
pickle: For serializing and deserializing Python objects.
r2_score: A function to calculate the R^2 (coefficient of determination) regression score.
GridSearchCV: A class for performing grid search with cross-validation to find the best hyperparameters for a model.
CustomException: Custom exception class, presumably for handling exceptions in a custom way.
'''
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException


'''This function saves a Python object to a file specified by file_path.
It first extracts the directory path from file_path using os.path.dirname.
Then, it creates the directory if it doesn't exist using os.makedirs.
Finally, it opens the file in binary write mode and uses pickle.dump to serialize and save the object to the file.
If any exception occurs during this process, it raises a CustomException.'''
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    


'''This function evaluates multiple models by training them on the provided training data (X_train, y_train) and then testing them on the provided testing data (X_test, y_test).
It uses grid search (GridSearchCV) to find the best hyperparameters (param) for each model.
For each model:
It initializes a grid search with cross-validation (GridSearchCV) using the model and parameter grid (param).
It fits the grid search on the training data to find the best parameters.
It sets the best parameters on the model using model.set_params.
It fits the model on the training data with the best parameters.
It predicts the target values for both training and testing data.
It calculates the R^2 score for both training and testing predictions.
It stores the test R^2 score in a dictionary (report) with the model name as the key.
If any exception occurs during this process, it raises a CustomException.'''


'''
This function takes several arguments:
X_train: Training features.
y_train: Target values for the training set.
X_test: Testing features.
y_test: Target values for the testing set.
models: A dictionary of models to evaluate.
param: A dictionary of hyperparameter grids for each model.

for i in range(len(list(models))):: This line iterates over each model in the models dictionary. len(list(models)) calculates the number of models in the dictionary, and range() creates a sequence of numbers from 0 to len(models) - 1, which corresponds to the indices of the models.
model = list(models.values())[i]: Inside the loop, this line extracts the model object at the current index i from the models dictionary. list(models.values()) gets a list of all the model objects, and [i] retrieves the model object at index i.
para=param[list(models.keys())[i]]: This line extracts the corresponding parameter dictionary for the current model from the param dictionary. list(models.keys()) gets a list of all the model names, and [i] retrieves the model name at index i. Then, this model name is used as a key to retrieve the parameter dictionary from the param dictionary.
gs = GridSearchCV(model, para, cv=3): Here, a GridSearchCV object gs is instantiated. GridSearchCV is a method for hyperparameter tuning. It takes the model (model), the parameter grid (para), and the number of cross-validation folds (cv=3) as arguments.
gs.fit(X_train, y_train): This line fits the GridSearchCV object gs to the training data (X_train and y_train). It performs an exhaustive search over the specified parameter grid (para) and selects the best parameters based on cross-validation performance.
model.set_params(**gs.best_params_): After the grid search, this line updates the parameters of the original model (model) with the best parameters found by the grid search. gs.best_params_ contains the best parameter values selected by the grid search.
model.fit(X_train, y_train): Finally, the model (model) is trained on the entire training data (X_train and y_train) using the updated parameters obtained from grid search.
Overall, this loop iterates over each model in the models dictionary, performs grid search cross-validation to tune hyperparameters, and then trains each model on the training data using the best parameters found by grid search.

'''
def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        '''Initializes an empty dictionary report to store the evaluation results of each model.'''
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            #model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)


'''This function loads a Python object from a file specified by file_path.
It opens the file in binary read mode and uses pickle.load to deserialize and load the object from the file.
If any exception occurs during this process, it raises a CustomException.'''   
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

'''

Grid search validation is a technique used for hyperparameter tuning in machine learning. It's a systematic method to find the best combination of hyperparameters for a given model by searching through a specified grid of hyperparameter values.

Here's how grid search validation works:

Define Hyperparameter Grid: First, you specify a grid of hyperparameters and their corresponding values that you want to search over. For example, if you're training a support vector machine (SVM) model, you might want to tune parameters like the kernel type, C (regularization parameter), and gamma.
Create a Search Space: Grid search creates a search space by considering all possible combinations of hyperparameters from the grid. Each combination represents a point in the search space.
Cross-Validation: For each combination of hyperparameters, grid search typically uses k-fold cross-validation to evaluate the model's performance. It divides the training data into k subsets (or folds), trains the model on k-1 folds, and evaluates it on the remaining fold. This process is repeated k times, and the average performance metric (e.g., accuracy, mean squared error) is computed across all folds.
Select Best Parameters: After evaluating the model's performance for each combination of hyperparameters, grid search selects the combination that produces the best performance according to a specified evaluation metric. This could be the highest accuracy, lowest error, etc.
Final Model Training: Finally, the model is trained using the entire training dataset with the best hyperparameters selected during the grid search process.
Grid search validation is beneficial because it automates the process of hyperparameter tuning, which can be tedious and time-consuming when done manually. It systematically explores the hyperparameter space and helps to find the optimal set of hyperparameters that yield the best performance for the model.
'''
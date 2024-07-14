'''
This code is a Flask application that serves a web page for predicting student exam performance based on various input features. Here's a breakdown of the code:

Import Statements:
Flask: This is the main Flask class used to create the web application.
request: It is used to access incoming request data in Flask.
render_template: This function is used to render HTML templates.
numpy (np): It is used for numerical computations.
pandas (pd): It is used for data manipulation and analysis.
StandardScaler: This class from scikit-learn is used for feature scaling.
CustomData and PredictPipeline: These are custom classes imported from the project's source code for handling data and prediction pipeline.
Flask Application Initialization:
An instance of the Flask class is created with the name application. This is the entry point of the Flask application.
app is set equal to application for convenience.
Route for Home Page ("/"):
@app.route('/') is a decorator that defines a route for the home page.
The index function is executed when a user visits the home page. It renders the index.html template.
Route for Predicting Data ("/predictdata"):
@app.route('/predictdata', methods=['GET', 'POST']) is a decorator that defines a route for predicting data.
The predict_datapoint function handles both GET and POST requests.
If the request method is GET, it renders the home.html template.
If the request method is POST, it collects form data submitted by the user, creates a CustomData object with the collected data, converts it into a pandas DataFrame, and then passes it to the PredictPipeline for prediction.
The prediction result is then passed to the home.html template for display.
Main Block:
The __name__ variable is checked to ensure that the script is being run directly and not imported as a module.
If the script is being run directly, the Flask application is started using app.run().
Overall, this Flask application serves a simple web page where users can input data related to a student's demographics and exam scores, and it provides a prediction for the student's performance.
'''

from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__) #entry point of the flask application

app=application

## Route for a home page

@app.route('/') #decorator to route the page 
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))

        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html',results=results[0])
    

if __name__=="__main__":
    app.run(host="0.0.0.0")  
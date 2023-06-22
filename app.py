import json
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd
import mlflow

from steps.data_processing import data_processing_function


app = Flask(__name__)

# Load the model
model_name = 'Loan_prediction'
stage = "Production"
model_registry_path = f'models:/{model_name}/{stage}'
production_model = mlflow.pyfunc.load_model(model_registry_path)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    raw_data = request.json['data']
    #data = json.dumps(data_processing_function(raw_data))
    output = production_model.predict(raw_data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict',methods=['POST'])
def predict():
    data = {}
    for key, value in request.form.items():
        data[key] = float(value)
    # data form adjustments
    print(data)
    output = production_model.predict(data)[0]
    return render_template("home.html",prediction_text="The predicted result is {}".format(output))

if __name__ == "__main__":
    app.run(debug=True)
    


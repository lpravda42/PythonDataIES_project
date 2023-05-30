# PythonDataIES_project
ML model predicting the probability of providing a loan

## Project Intro/Objective
The project aims to build a machine learning model that predicts the probability of a customer being approved for a loan based on their specific characteristics. This model is designed to be used directly by the customer to assess their eligibility for a loan before they apply. By inputting their relevant data into the model, the customer will receive a personalized probability score indicating their likelihood of a loan approval. Based on a Kaggle dataset.

### Methods Used
* Data Visualization
* etc.

### Technologies
* Python
* Pandas, jupyter
* etc. 

## Progress

The current main python script is 'train.py'. The project environment is specified in 'MLproject' and 'conda.yaml'. You can install the conda environment by running 
```bash
conda env create -f conda.yaml
```
and activate it with
```bash
conda activate Loan_prediction
```

'train.py' can be run inside the terminal using command 
```bash
python train.py {max_iter}
```

Model metrics for a given parameter 'max_iter' will be displayed. The model and its results for a given run can be found online, you need to type 
```bash
mlflow ui 
```
inside a terminal and open the resulting http adress   


## Needs to be done

The original data set is used at the moment so the data processing and analysis should be created, packaged and run in the main training file to load the data.

Only logistic regression with one parameter is employed, other models and parameters will be added.

Metric for final model selection and further registration should follow after this code, the best-performing model then will be logged and put on the server

The project ends with model deployment, the idea is a web form predicting the probability after filling the independent variables.

# Import libraries
import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from urllib.parse import urlparse
import mlflow
from mlflow.models.signature import infer_signature
import mlflow.sklearn

import logging

# Set logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


# Define evaluation matrics
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(8)
    
    # Load data
    try:
        data = pd.read_csv(f'train_u6lujuX_CVtuZ9i.csv')
    except Exception as e:
        logger.exception(
            "Unable to download the data set. Error %s", e
        )
    
    # Data adjustments - will be deleted
    data = data.drop(['Loan_ID'], axis=1)
    for i in ["Gender", "Married", "Dependents", "Self_Employed", "Loan_Amount_Term", "Credit_History"]:
        data[i].fillna(data[i].mode()[0], inplace = True)
    data["LoanAmount"].fillna(data["LoanAmount"].mean(), inplace = True)
    data = pd.get_dummies(data, drop_first=True)


    # Split data into training and testing samples
    X = data.drop(["Loan_Status_Y"], axis=1)
    y = data["Loan_Status_Y"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=9)

    # Model parameters
    max_iter = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    
    
    # Run the Logistic regression
    with mlflow.start_run():
        logistic_reg = LogisticRegression(solver="liblinear", max_iter=max_iter, random_state=10)
        logistic_reg.fit(X_train, y_train)
    
        y_pred = logistic_reg.predict(X_test)
         
        (rmse, mae, r2) = eval_metrics(y_test, y_pred)        
        
        # Print model metrics
        print("Logistic model (max_iter = {:f}):".format(max_iter))
        print("RMSE: %s" % rmse)
        print("MAE: %s" % mae)
        print("R2: %s" % r2)
        
        # Log parameters
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        
        predictions = logistic_reg.predict(X_train)
        signature = infer_signature(X_train, predictions)
        
        # Register model
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(
                logistic_reg, "Logit", registered_model_name="LogisticRegression", signature=signature
            )
        else:
            mlflow.sklearn.log_model(logistic_reg, "Logit", signature=signature)
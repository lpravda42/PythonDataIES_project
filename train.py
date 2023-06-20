# Import libraries
import os
import warnings
import sys
import json

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from urllib.parse import urlparse
import mlflow
from mlflow.models.signature import infer_signature
import mlflow.sklearn
from mlflow.tracking import MlflowClient

import logging
import argparse

from steps.data_processing import data_processing_function

# Creating experiment on remote server
#mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Loan_prediction")

# Set logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# Console command
parser = argparse.ArgumentParser(description="Model training parameters")

logit_parser = parser.add_argument_group("Logistic regression")
logit_parser.add_argument('--logit-max_iter', type=int, default=100, help='Maximum number of iterations')
logit_parser.add_argument('--logit-penalty', choices=['l1','l2','elasticnet',"None"], default='l2', help='Norm of the penalty')
logit_parser.add_argument('--logit-solver', choices=['lbfgs','liblinear','newton-cholesky','sag','saga'], default='lbfgs', help='Algorithm for optimization')
logit_parser.add_argument('--logit-C', type=float, default=1.0, help='Regularization strength')

rf_parser = parser.add_argument_group("Random forest")
rf_parser.add_argument('--rf-n_estimators', type=int, default=100, help='Number of decision trees in the forest')
rf_parser.add_argument('--rf-max_leaf_nodes', type=int, default=None, help='Maximum number of leaf nodes')
rf_parser.add_argument('--rf-criterion', choices=['gini','entropy','log_loss'], default='gini', help='Quality of a split measure')
rf_parser.add_argument('--rf-max_depth', type=int, default=None, help='Maximum depth of the tree')

metric_parser = parser.add_argument_group("Evaluation metrics")
metric_parser.add_argument('--metric', choices=['rmse','mae','r2'], default='rmse', help='Metric used for final model selection')

args = parser.parse_args()

logit_max_iter = args.logit_max_iter
logit_penalty = args.logit_penalty
logit_solver = args.logit_solver
logit_C = args.logit_C

rf_n_estimators = args.rf_n_estimators
rf_max_leaf_nodes = args.rf_max_leaf_nodes
rf_criterion = args.rf_criterion
rf_max_depth = args.rf_max_depth

metric = args.metric

# Define evaluation matrics
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

# Model training
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(8)
    
    # Load data
    try:
        raw_data = pd.read_csv(f'train_u6lujuX_CVtuZ9i.csv')
    except Exception as e:
        logger.exception(
            "Unable to download the data set. Error %s", e
        )
    
    # Data adjustments
    data = data_processing_function(raw_data)

    # Split data into training and testing samples
    X = data.drop(["Loan_Status_Y"], axis=1)
    y = data["Loan_Status_Y"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=9)
    

    # Logistic regression
    with mlflow.start_run(run_name="Logistic regression"):
        logit_params = {
            'penalty': logit_penalty if logit_penalty != "None" else None,
            'C': logit_C,
            'solver': logit_solver,
            'max_iter': logit_max_iter,
            'random_state': 10
        }
        
        logit = LogisticRegression(**logit_params)
        logit.fit(X_train, y_train)
        y_pred_logit = logit.predict(X_test)
         
        (rmse_logit, mae_logit, r2_logit) = eval_metrics(y_test, y_pred_logit)        
        
        # Print model metrics
        if logit_penalty is None:
            logit_penalty_str = "None"
        else: logit_penalty_str = str(logit_penalty)
        
        print("Logistic regression (penalty={:s}, C={:f}, solver={:s}, "
              "max_iter={:f}):".format(logit_penalty_str,logit_C,logit_solver,logit_max_iter))
        print("RMSE: %s" % rmse_logit)
        print("MAE: %s" % mae_logit)
        print("R2: %s" % r2_logit)
        print()
        
        # Log parameters
        mlflow.set_tag("model_name","logit")
        mlflow.log_params(logit_params)
        mlflow.log_metric("rmse", rmse_logit)
        mlflow.log_metric("mae", mae_logit)
        mlflow.log_metric("r2", r2_logit)
        
        # Log model
        fitted_logit = logit.predict(X_train)
        signature_logit = infer_signature(X_train, fitted_logit) 
        
        mlflow.sklearn.log_model(
                logit, "logit", signature=signature_logit
            )
        
        
    # Random forest
    with mlflow.start_run(run_name="Random forest"):
        rf_params = {
            'n_estimators': rf_n_estimators,
            'max_leaf_nodes': rf_max_leaf_nodes,
            'criterion': rf_criterion,
            'max_depth': rf_max_depth,
            'random_state': 1
        }
        
        rf = RandomForestClassifier(**rf_params)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        
        (rmse_rf, mae_rf, r2_rf) = eval_metrics(y_test, y_pred_rf)
        
        # Print model metrics
        if rf_max_leaf_nodes is None:
            rf_max_leaf_nodes_str = "None"
        else: rf_max_leaf_nodes_str = str(rf_max_leaf_nodes)
        if rf_max_depth is None:
            rf_max_depth_str = "None"
        else: rf_max_depth_str = str(rf_max_depth)
        
        print("Random forest (n_arguments={:f}, max_leaf_nodes={:s}, criterion={:s}, "
              "max_depth={:s}):".format(rf_n_estimators,rf_max_leaf_nodes_str,rf_criterion,rf_max_depth_str))
        print("RMSE: %s" % rmse_rf)
        print("MAE: %s" % mae_rf)
        print("R2: %s" % r2_rf)
        print()
        
        # Log parameters
        mlflow.set_tag("model_name","rf")
        mlflow.log_params(rf_params)
        mlflow.log_metric("rmse", rmse_rf)
        mlflow.log_metric("mae", mae_rf)
        mlflow.log_metric("r2", r2_rf)
        
        
        # Log model
        fitted_rf = rf.predict(X_train)
        signature_rf = infer_signature(X_train, fitted_rf) 
        
        mlflow.sklearn.log_model(
                rf, "rf", signature=signature_rf
            )


    ## Select the best model ##
    # Get experiment ID
    current_experiment = dict(mlflow.get_experiment_by_name("Loan_prediction"))
    experiment_id = current_experiment['experiment_id']
    
    # Work with MLflowClinet to access model elements 
    client = MlflowClient()
    
    # Search for runs in a Loan_prediction experiment and select the best-performing run based on chosen metric
    if metric == "rmse":
        runs = client.search_runs(
            [experiment_id],
            order_by = ["metrics.rmse"]
        )
        best_run = np.argmin([f.data.metrics['rmse'] for f in runs])
        best_rmse = np.round(runs[best_run].data.metrics['rmse'],4)
        print(f"Experiment has {len(runs)} runs")
        print(f"Best run - {best_run} with rmse of {best_rmse}")
    elif metric == "mae":
        runs = client.search_runs(
            [experiment_id],
            order_by = ["metrics.mae"]
        )
        best_run = np.argmin([f.data.metrics['mae'] for f in runs])
        best_mae = np.round(runs[best_run].data.metrics['mae'],4)
        print(f"Experiment has {len(runs)} runs")
        print(f"Best run - {best_run} with mae of {best_mae}")
    else:    
        runs = client.search_runs(
            [experiment_id],
            order_by = ["metrics.r2 DESC"]
        )
        best_run = np.argmax([f.data.metrics['r2'] for f in runs])
        best_r2 = np.round(runs[best_run].data.metrics['r2'],4)
        print(f"Experiment has {len(runs)} runs")
        print(f"Best run - {best_run} with r2 of {best_r2}")
    
    
    # "jsonify" log-model history
    log_model_info = json.loads(runs[best_run].data.tags['mlflow.log-model.history'])[0]
    
    # Construct model URI
    model_uri = 'runs:/' + log_model_info['run_id'] + '/' + log_model_info['artifact_path']
    print(f"Best model URI: {model_uri}")
    
    # Register model
    model_name = 'Loan_prediction'
    model_version = 1
    
    mlflow.register_model(model_uri, model_name)
    
    # Promote to Production
    logs = client.transition_model_version_stage(name=model_name,version=model_version,stage="Production")
    
    # Predictions
    stage = "Production"
    model_registry_path = f'models:/{model_name}/{stage}'
    production_model = mlflow.pyfunc.load_model(model_registry_path)

    prediction = production_model.predict(X_test[:1])
    
    print(f"Model prediction: {prediction}")
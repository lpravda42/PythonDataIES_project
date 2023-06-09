import os
import argparse

from hyperopt import fmin, hp, tpe
from hyperopt.pyll import scope

import mlflow.projects
from mlflow.tracking import MlflowClient

# Console command
parser = argparse.ArgumentParser(description = "Perform hyperparameter search")

parser.add_argument('--max-runs', type=int, default=10, help="Maximum number of runs to evaluate")
parser.add_argument('--model-type', choices=['logit','rf'], default='logit', help="Model type to tune")

args = parser.parse_args()
max_runs = args.max_runs
model_type = args.model_type

print(max_runs)
print(model_type)

# Hyperparameter search
def train(max_runs, model_type):
    """
    Run hyperparameter optimization
    """
    tracking_client = MlflowClient()
    
    def new_eval(experiment_id):
        """_summary_

        Args:
            ecperiment_id (_type_): _description_
        """
        def eval(params):
            """_summary_

            Args:
                params (_type_): _description_
            """
            with mlflow.start_run(nested=True) as child_run:
                if model_type == "logit":
                    # Params for logit
                    (
                        penalty,C,solver,max_iter
                        ) = params 
                    # Run train.py as MLflow sub-run
                    p = mlflow.projects.run(
                        uri=".", # current directory
                        entry_point="train",
                        run_id=child_run.info.run_id,
                        parameters={
                            "penalty": penalty,
                            "C": C,
                            "solver": solver,
                            "max_iter": max_iter
                        },
                        experiment_id=experiment_id,
                        synchronous=False
                    )
                    succeeded = p.wait()
                    # Log parameters
                    mlflow.log_params(
                        {
                            "penalty": penalty,
                            "C": C,
                            "solver": solver,
                            "max_iter": max_iter
                        }
                    )
                elif model_type == "rf":
                    # Params for random forest
                    (
                        n_estimators,max_leaf_nodes,criterion,max_depth
                        ) = params
                    # Run train.py as MLflow sub-run
                    p = mlflow.projects.run(
                        uri=".",
                        entry_point="train",
                        run_id=child_run.info.run_id,
                        parameters={
                            "n_estimators": n_estimators,
                            "max_leaf_nodes": max_leaf_nodes,
                            "criterion": criterion,
                            "max_depth": max_depth 
                        },
                        experiment_id=experiment_id,
                        synchronous=False
                    )
                    succeeded = p.wait()
                    mlflow.log_params(
                        {
                            "n_estimators": n_estimators,
                            "max_leaf_nodes": max_leaf_nodes,
                            "criterion": criterion,
                            "max_depth": max_depth
                        }
                    )
                    print(succeeded) # wait for the run to finish and print if it succeeded
            # Save metrics from MLflow run
            training_run = tracking_client.get_run(p.run_id)
            metrics = training_run.data.metrics
            test_rmse = metrics["rmse"]                       
            
            return test_rmse
        
        return eval
                
    # Definition of space
    if model_type == "logit":
        # Search space for Logistic regression
        space = [
            hp.choice('penalty',['l1','l2','elasticnet',None]),
            hp.loguniform('C',low=-3,high=3),
            hp.choice('solver',['lbfgs','liblinear','newton-cholesky','sag','saga']),
            scope.int(hp.quniform('max_iter',low=50,high=500,q=10))
        ]
    elif model_type == "rf":
        # Search space for Random forest
        space = [
            scope.int(hp.quniform('n_estimators',low=50,high=1000,q=50)),
            hp.choice('max_leaf_nodes',[None,scope.int(hp.quniform('max_leaf_nodes_value',low=10,high=100,q=10))]),
            hp.choice('criterion',['gini','entropy','log_loss']),
            hp.choice('max_depth',[None,scope.int(hp.quniform('max_depth_value',low=1,high=10,q=1))])
        ]
    else:
        raise ValueError(f"Model type {model_type} is not supported")
    
            
    # Actual search.py experiment run
    with mlflow.start_run() as run:
        # Get parent ID
        experiment_id = run.info.experiment_id
        
        # Optimization function
        best = fmin(
            fn=new_eval(experiment_id),
            space=space,
            algo=tpe.suggest,
            max_evals=max_runs
        )
        mlflow.set_tag("best_params",str(best))


if __name__ == "__main__":
    train(max_runs, model_type)
import mlflow
import mlflow.sklearn
import os

def init_experiment(experiment_name="AutoML_Pipeline"):
    mlflow.set_tracking_uri("file://"+os.path.abspath("mlruns"))
    mlflow.set_experiment(experiment_name)

def start_run(run_name=None):
    return mlflow.start_run(run_name=run_name)

def log_params(metrics:dict):
    for key,value in metrics.items():
        mlflow.log_param(key,value)

def log_metrics(metrics:dict):
    for key,value in metrics.items():
        mlflow.log_metric(key,value)

def log_model(model,model_name="model"):
    mlflow.sklearn.log_model(model,model_name)

def end_run():
    mlflow.end_run()

import mlflow
import string
import random

def get_random_string(length: int):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    print("Random string of length", length, "is:", result_str)
    return result_str


def set_mlflow(config_file: str=None, 
               experiment_id:str=None,
               experiment_name:str=None,
               run_name:str=None):
    """Set MLflow tracking."""

    # Set MLflow tracking
    # Create an experiemnt
    # Set the experiment via environment variables
    # export MLFLOW_EXPERIMENT_NAME=fraud-detection
    # mlflow experiments create --experiment-name covid-xrays-clf_comp-methods        
    if not experiment_id:
        # Create experiment
        experiment_id = mlflow.create_experiment(experiment_name)
        experiment = mlflow.get_experiment(experiment_id)
        
        #experiment = mlflow.set_experiment(param['mlflow']['experiment_name'])
    if run_name:
        # Combine given run name with a random string
        run_name = f"{run_name}_{get_random_string(3)}"
        if experiment_name:
            mlflow.start_run(run_name=run_name, experiment_id=experiment_id)
        else:
            mlflow.start_run(run_name=run_name)

    # mlflow autolog
    mlflow.tensorflow.autolog()

    # Log config file as artifact
    mlflow.log_artifact(config_file, artifact_path='config_file')

def stop_mlflow():
    """Stop MLflow tracking."""
    mlflow.end_run()
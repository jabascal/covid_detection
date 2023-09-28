import mlflow
from utils.helper_inout import get_random_string

def set_mlflow(config_file: str=None, 
               experiment_id:str=None,
               experiment_name:str=None,
               run_name:str=None, 
               every_n_iter:int=1):
    """Set MLflow tracking."""

    # Set MLflow tracking
    # Create an experiemnt
    # Set the experiment via environment variables
    # export MLFLOW_EXPERIMENT_NAME=fraud-detection
    # mlflow experiments create --experiment-name covid-xrays-clf_comp-methods        
    if not experiment_id:
        try: 
            # Load if experiment exist by name
            experiment = mlflow.get_experiment_by_name(experiment_name)
            experiment_id = experiment.experiment_id
        except:
            # Create experiment
            experiment_id = mlflow.create_experiment(experiment_name)
          # experiment = mlflow.get_experiment(experiment_id)        
    if run_name:
        # Combine given run name with a random string
        run_name = f"{run_name}_{get_random_string(3)}"
        if experiment_name:
            mlflow.start_run(run_name=run_name, experiment_id=experiment_id)
        else:
            mlflow.start_run(run_name=run_name)

    # mlflow autolog
    mlflow.tensorflow.autolog(every_n_iter=every_n_iter)

    # Log config file as artifact
    mlflow.log_artifact(config_file, artifact_path='config_file')

def save_model_mlflow(model, model_name: str):
    # Save model artifact for MLflow
    mlflow.tensorflow.log_model(model, model_name)

def log_figure_mlflow(fig, artifact_file: str):
    # Log figure as artifact
    mlflow.log_figure(fig, artifact_file)

def stop_mlflow():
    """Stop MLflow tracking."""
    mlflow.end_run()

"""
# Save model

# Infer signature
from mlflow.models.signature import infer_signature
model_signature = infer_signature(imgs_val, model.predict(imgs_val))

# Save model artifact for MLflow
mlflow.tensorflow.log_model(model, "model", signature=model_signature)

"""
"""
Train and fine-tune a classifier model on a covid dataset, with experiment tracking using Tensorboard and mlflow.
Call the function train_finetune_clf() 
"""
import argparse
import random
import os
from utils.helper_config import load_config, set_params_debug
from utils.helper_tf import train_finetune_clf 
from utils.helper_mlflow import set_mlflow, stop_mlflow

random.seed(123)

# Debug mode
debug_mode = False

# ---------------------------------------------
# Parse config file
parser = argparse.ArgumentParser()
parser.add_argument('--config', default='config/config_clf.yaml', help='YAML config file path')
args = parser.parse_args()
print(f'Args: {args}')

print(f'Config file: {args.config}')

# Laod config file
param = load_config(args.config)
#----------------------------------------------
# Rewrite parameters (for tests and hyperparameter tuning)
param['data']['train_path'] = '../../../Data/COVID-19_CXR_Dataset_final_train'
param['data']['test_path'] = '../../../Data/COVID-19_CXR_Dataset_final_test'

if debug_mode:
    param = set_params_debug(param)

# Rewrite parameters for current run
if False:
    param['train']['continue_training'] = True
    param['data']['cache'] = False
    param['data']['train_size'] = 2
    param['data']['val_size'] = 2
    param['data']['test_size'] = 2    

# ---------------------------------------------
# Train and fine-tune a classifier model
if __name__ == "__main__":
    # ---------------------------------------------
    # Set MLflow tracking
    if param['mlflow']['tracking']:
        set_mlflow(config_file=args.config, 
                   experiment_id=param['mlflow']['experiment_id'],
                   experiment_name=param['mlflow']['experiment_name'],
                   run_name=param['mlflow']['run_name'])
    # ---------------------------------------------
    # Train and fine-tune a classifier model
    model, history, test_loss, test_acc = train_finetune_clf(
            # Data
            train_dir=param['data']['train_path'],
            test_dir=param['data']['test_path'],
            val_dir=param['data']['val_path'],
            img_height=param['data']['img_height'],
            img_width=param['data']['img_width'],
            batch_size=param['train']['batch_size'],
            validation_split=param['data']['val_split'],
            test_split=param['data']['test_split'],
            color_mode=param['data']['color'],
            augmentation_param=param['data']['augmentation'],
            cache=param['data']['cache'],
            shuffle=param['data']['shuffle'],
            train_size=param['data']['train_size'],
            val_size=param['data']['val_size'],
            test_size=param['data']['test_size'],
            # Model
            base_model_name=param['model']['base_model_name'],
            model_num_channels=param['model']['num_channels'],
            dropout=param['model']['dropout'],
            path_save_model=param['model']['path_save'],
            # Train
            continue_training=param['train']['continue_training'],
            initial_epochs=param['train']['epochs'],
            fine_tune_at_perc=param['train']['fine_tune_at_perc'],
            base_learning_rate=param['train']['lr'],
            fine_tune_epochs=param['train']['epochs_finetune'],
            ft_learning_rate=param['train']['lr_finetune'],
            metrics=param['train']['metrics'],
            mode_display=param['train']['mode_display'],
            # Tensorboard
            log_dir=param['tb']['log_dir'],
            histogram_freq=param['tb']['histogram_freq'],
            profile_batch=param['tb']['profile_batch'],
            # Early stopping
            early_stopping_patience=param['tb']['early_stopping']['patience'],
            early_stopping_monitor=param['tb']['early_stopping']['monitor'],
            # Model checkpoint
            ckpt_freq=param['tb']['model_ckpt']['ckpt_freq'],
            ckpt_path=param['tb']['model_ckpt']['ckpt_path'],
            ckpt_monitor=param['tb']['model_ckpt']['ckpt_monitor'],
            # Reduce learning rate
            reduce_lr_monitor=param['tb']['reduce_lr']['monitor'],
            reduce_lr_factor=param['tb']['reduce_lr']['factor'],
            reduce_lr_patience=param['tb']['reduce_lr']['patience'],
            reduce_lr_min=param['tb']['reduce_lr']['min_lr'],
            # Config file
            config_file=args.config,
            # mlflow
            mlflow_exp=param['mlflow']['tracking'],
            path_model_prev=param['mlflow']['path_model_prev'],
    )
    # ---------------------------------------------
    # Save model artifact for MLflow
    #mlflow.tensorflow.log_model(model, "model")

    # ---------------------------------------------
    # Stop mlflow
    if param['mlflow']['tracking']:
        stop_mlflow()



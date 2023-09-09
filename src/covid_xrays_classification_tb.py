"""
Train and fine-tune a classifier model on a covid dataset, with experiment tracking using Tensorboard.
Call the function train_finetune_clf() 
"""
import argparse
import random

from utils.helper_in_out import load_config
from utils.helper_tf import train_finetune_clf 

random.seed(123)

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
param['data']['path'] = '../../../Data/COVID-19_CXR_Dataset_final'
param['train']['epochs'] = 1
param['train']['epochs_finetune'] = 1
# ---------------------------------------------
# Train and fine-tune a classifier model
if __name__ == "__main__":
    model, history, test_loss, test_acc = train_finetune_clf(
                        # Data
                        data_dir=param['data']['path'],
                        img_height=param['data']['img_height'],
                        img_width=param['data']['img_width'],
                        batch_size=param['train']['batch_size'],
                        validation_split=param['data']['val_split'],
                        test_split=param['data']['test_split'],
                        color_mode=param['data']['color'],
                        augmentation_param=param['data']['augmentation'],
                        cache=param['data']['cache'],
                        shuffle=param['data']['shuffle'],
                        # Model
                        base_model_name=param['model']['base_model_name'],
                        model_num_channels=param['model']['num_channels'],
                        dropout=param['model']['dropout'],
                        # Train
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
                        reduce_lr_min=param['tb']['reduce_lr']['min_lr']
    )


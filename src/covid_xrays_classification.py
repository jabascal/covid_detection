"""
Call the function train_finetune_clf() to train and fine-tune a classifier model.
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
# ---------------------------------------------
# Train and fine-tune a classifier model
if __name__ == "__main__":
    train_finetune_clf(data_dir=param['data']['path'],
                        img_height=param['data']['img_height'],
                        img_width=param['data']['img_width'],
                        batch_size=param['train']['batch_size'],
                        validation_split=param['data']['val_split'],
                        test_split=param['data']['test_split'],
                        color_mode=param['data']['color'],
                        augmentation_param=param['data']['augmentation'],
                        cache=param['data']['cache'],
                        shuffle=param['data']['shuffle'],
                        #
                        base_model_name=param['model']['base_model_name'],
                        model_num_channels=param['model']['num_channels'],
                        dropout=param['model']['dropout'],
                        #
                        initial_epochs=param['train']['epochs'],
                        fine_tune_at_perc=param['train']['fine_tune_at_perc'],
                        base_learning_rate=param['train']['lr'],
                        fine_tune_epochs=param['train']['epochs_finetune'],
                        ft_learning_rate=param['train']['lr_finetune'],
                        metrics=param['train']['metrics'],
                        mode_display=param['train']['mode_display'],
    )

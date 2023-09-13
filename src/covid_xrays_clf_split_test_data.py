"""
Split data into train and test sets and move the test data into a new directory
and rename the source data directory as train directory
"""

from utils.helper_inout import split_move_data_train_test


# Dataset path
data_dir = '../../../Data/COVID-19_CXR_Dataset_final'
test_dir = '../../../Data/COVID-19_CXR_Dataset_final_test'
test_percent = 10
split_move_data_train_test(data_dir=data_dir, test_dir=test_dir, test_percent=test_percent)
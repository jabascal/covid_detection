import os
import random
import shutil
random.seed(123)

def split_move_data_train_test(data_dir: str, test_dir: str, test_percent: int):
    """
    Split data into train and test sets and move the test data into a new directory
    and rename the source data directory as train directory
    """

    # Create the test directory if it doesn't exist
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    # Loop through each subfolder in the data directory
    for root, dirs, files in os.walk(data_dir):
        for subfolder in dirs:
            subfolder_path = os.path.join(root, subfolder)
            # Create the test subdirectory if it doesn't exist
            test_subfolder_path = os.path.join(test_dir, os.path.relpath(subfolder_path, data_dir))
            if not os.path.exists(test_subfolder_path):
                os.makedirs(test_subfolder_path)
            
            # Get a list of all the image files in the subfolder
            image_files = []
            for filename in os.listdir(subfolder_path):
                if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
                    image_files.append(filename)
            
            # Shuffle the image files
            random.shuffle(image_files)
            
            # Calculate the number of images to use for testing
            num_test_images = int(len(image_files) * (test_percent / 100))
            
            # Move the test images to the test directory
            for i in range(num_test_images):
                src_path = os.path.join(subfolder_path, image_files[i])
                dst_path = os.path.join(test_subfolder_path, image_files[i])
                shutil.move(src_path, dst_path)
    
    # Change data_dir to train_dir
    train_dir = f'{data_dir}_train'
    os.rename(data_dir, train_dir)
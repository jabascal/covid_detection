import  PIL
import pathlib
import random

from utils.helper_plot import plot_grid_images_from_file, plot_histogram

def get_images_size(files:list, num_images:int=500):
    """Get images size from list of files"""
    # Read image size
    imgs_shape = []
    imgs_height = []
    for file in files[:num_images]:
        img_size = PIL.Image.open(file).size
        imgs_shape.append(img_size)
        imgs_height.append(img_size[1])
    return imgs_shape, imgs_height

def inspect_data_plot(data_dir:str, mode_display:bool=False):
    """Inspect the data"""

    # Inspect the data
    files = list(pathlib.Path(data_dir).glob('*/*.png'))
    files = [str(path) for path in files]
    random.shuffle(files)

    # Print some random images
    if mode_display:
        plot_grid_images_from_file(files, n_images=9)

    # Count the number of images in the dataset
    img_count = len(files)
    print(f"Number of images: {img_count}")
    return files, img_count

def inspect_images_size(files:list, mode_display:bool=False):
    """Inspect the images size"""
    # Read image size
    imgs_shape, imgs_height = get_images_size(files, num_images=500)

    if mode_display:
        plot_histogram(imgs_height, title="Image height distribution", xlabel="Height")
    return imgs_shape, imgs_height
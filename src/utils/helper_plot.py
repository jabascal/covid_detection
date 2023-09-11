import numpy as np
import matplotlib.pyplot as plt
import PIL
import itertools

def plot_grid_images_from_file(files:list, 
                               n_images:int=9, 
                               figsize:tuple=(10, 10)):
    """Plot grid of images from list of files"""
    plt.figure(figsize=figsize)
    for i, file in enumerate(files[:n_images]):
        ax = plt.subplot(3, 3, i + 1)
        img = PIL.Image.open(file)
        print(f"Image shape: {img.size}")
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)

def plot_grid_images_from_array(imgs:np.ndarray,                                  
                                imgs_titles:list,
                                cmap:str='gray',
                                vmin:int=0,
                                vmax:int=255,
                                figsize:tuple=(10, 10)):
    """Plot grid of images from ndarray"""
    num_images = min(9, len(imgs))
    plt.figure(figsize=figsize)
    for i in range(num_images):
        print(f"Image shape: {imgs[i].shape}")
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(imgs[i], cmap=cmap, vmin=vmin, vmax=vmax)
        plt.title(imgs_titles[i])
        plt.colorbar()


def plot_histogram(values:list, 
                   title=None, 
                   xlabel=None, 
                   ylabel='Counts', 
                   bins:int=20):
    """Plot histogram"""
    fig = plt.figure()
    plt.hist(values, bins=bins)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    if xlabel:    
        plt.xlabel(xlabel)

def plot_train_curves_two_figs(fig1_values: list, 
                               fig1_legends: list,
                               fig1_title: str,
                               fig1_ylabel: str,
                               fig2_values: list,
                               fig2_legends: list,
                               fig2_title: str,
                               fig2_ylabel: str,
                               fig1_loc: str='lower right',
                               fig2_loc: str='upper right',
                               xlabel: str='epochs',
                               figsize: tuple=(10, 10)):
    plt.figure(figsize=figsize)
    plt.subplot(2, 1, 1)
    for i, values in enumerate(fig1_values):
        plt.plot(values, label=fig1_legends[i])
    plt.legend(loc=fig1_loc)
    plt.ylabel(fig1_ylabel)
    plt.xlabel(xlabel)
    plt.title(fig1_title)

    plt.subplot(2, 1, 2)
    for i, values in enumerate(fig2_values):
        plt.plot(values, label=fig2_legends[i])
    plt.legend(loc=fig2_loc)
    plt.ylabel(fig2_ylabel)
    plt.xlabel(xlabel)

def plot_loss_acc(loss: list, val_loss: list, acc: list, val_acc: list):
    """Plot loss and accuracy"""
    plot_train_curves_two_figs([acc, val_acc], 
                            ['Training Accuracy', 'Validation Accuracy'],
                            'Training and Validation Accuracy',
                            'Accuracy',
                            [loss, val_loss],
                            ['Training Loss', 'Validation Loss'],
                            'Training and Validation Loss',
                            'Loss')

def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
        cm (array, shape = [n, n]): a confusion matrix of integer classes
        class_names (array, shape = [n]): String names of the integer classes
    """
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Compute the labels from the normalized confusion matrix.
    labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure
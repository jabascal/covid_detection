import os
import pickle
import tensorflow as tf
from tensorflow.python.client import device_lib

from utils.helper_plot import plot_grid_images_from_array
from utils.helper_plot import plot_loss_acc

def get_prep_input_layer(model_name: str) -> tf.keras.layers:
    """Get the preprocess input layer for a given model name."""
    if model_name == 'mobilenet_v2':
        prep_input_layer = tf.keras.applications.mobilenet_v2.preprocess_input
    elif model_name == 'resnet50':
        prep_input_layer = tf.keras.applications.resnet50.preprocess_input
    elif model_name == 'inception_v3':
        prep_input_layer = tf.keras.applications.inception_v3.preprocess_input
    elif model_name == 'xception':
        prep_input_layer = tf.keras.applications.xception.preprocess_input
    elif model_name == 'vgg19':
        prep_input_layer = tf.keras.applications.vgg19.preprocess_input
    return prep_input_layer


def get_base_model(model_name: str, img_shape: tuple) -> tf.keras.Model:
    """Get the base model for a given model name."""
    if model_name == 'mobilenet_v2':
        base_model = tf.keras.applications.MobileNetV2(input_shape=img_shape,
                                                include_top=False,
                                                weights='imagenet')
    elif model_name == 'resnet50':
        base_model = tf.keras.applications.ResNet50(input_shape=img_shape,
                                                include_top=False,
                                                weights='imagenet')
    elif model_name == 'inception_v3':
        base_model = tf.keras.applications.InceptionV3(input_shape=img_shape,
                                                include_top=False,
                                                weights='imagenet')
    elif model_name == 'xception':
        base_model = tf.keras.applications.Xception(input_shape=img_shape,
                                                include_top=False,
                                                weights='imagenet')
    elif model_name == 'vgg19':
        base_model = tf.keras.applications.VGG19(input_shape=img_shape,
                                                include_top=False,
                                                weights='imagenet')
    return base_model

def get_data_augmentation_layer(param: dict) -> tf.keras.Model:
    data_augmentation_layer = tf.keras.Sequential([
        tf.keras.layers.RandomFlip(param['flip']),
        tf.keras.layers.RandomRotation(param['rotation']),
        ], name='data_augmentation')
    return data_augmentation_layer

def get_model_from_base(model_name: str, 
                           img_shape: tuple, 
                           num_classes: int, 
                           augmentation_param: dict,
                           trainable: bool=False,
                           dropout: float=0.2) -> tf.keras.Model:
    """Create a model from a base model."""

    # Data augmentation
    if augmentation_param:
        data_augmentation_layer = get_data_augmentation_layer(augmentation_param)

    # Preprocess input
    prep_input_layer = get_prep_input_layer(model_name)

    # Pre-trained model 
    # keep the BatchNormalization layers in inference mode by passing training = False during call
    base_model = get_base_model(model_name, img_shape)
    if trainable is False:
        base_model.trainable = False
    print('Base model summary:')
    base_model.summary()

    #image_batch, label_batch = next(iter(train_ds))
    #feature_batch = base_model(image_batch)
    #print(feature_batch.shape)

    # Add a classification head
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D(name='global_average')
    #feature_batch_average = global_average_layer(feature_batch)
    #print(feature_batch_average.shape)

    prediction_layer = tf.keras.layers.Dense(num_classes, name='prediction')

    # Build the model
    inputs = tf.keras.Input(shape=img_shape)
    if augmentation_param:
        x = data_augmentation_layer(inputs)
    x = prep_input_layer(x)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)

    print('Model summary:')
    model.summary()

    #tf.keras.utils.plot_model(model, show_shapes=True)

    return model, base_model

def unfreeze_model(base_model: tf.keras.Model, 
                   model: tf.keras.Model,
                   fine_tune_at: int):
    """Unfreeze the model for finetuning."""

    # Un-freeze the top layers of the model
    base_model.trainable = True
    print(f"Number of layers in the base model: {len(base_model.layers)}")

    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    model.summary()
    print(f"Number trainable layers: {len(model.trainable_variables)}")

    return base_model, model

def get_prediction(model: tf.keras.Model, 
                   img: tf.Tensor, 
                   class_names: list) -> tuple:
    """
    Get the prediction from a model.
        Args:
            model: tf.keras.Model
            img: tf.Tensor of dimension (height, width, channels)
    """
    
    predictions = model.predict_on_batch(tf.expand_dims(img, axis=0))
    score = tf.nn.softmax(predictions[0])    
    class_idx = tf.argmax(score)
    pred_name = class_names[class_idx]
    return pred_name, score[class_idx].numpy()

def get_predictions(model: tf.keras.Model,
                    imgs: list,
                    class_names: list) -> list:
    """
    Get the predictions from a model.
        Args:
            model: tf.keras.Model
            imgs: tf.Tensor of dimension (batch_size, height, width, channels)
    """
    preds_name = []
    preds_score = []
    for img in imgs:        
        pred_name, score = get_prediction(model, img, class_names)
        preds_name.append(pred_name)
        preds_score.append(score)
    return pred_name, preds_score

def create_dataset(data_dir: str,
                   img_width: int,
                   img_height: int,
                   batch_size: int,
                   validation_split: float,
                   test_split: int,
                   color_mode: str,
                   cache: bool=False,
                   shuffle: bool=True,
                   mode_display: bool=False):
    """Create a dataset from a directory."""
    
    AUTOTUNE = tf.data.AUTOTUNE

    # Create data set 
    # Split in training and validation batched dataset
    train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        seed=123,
        shuffle=True,
        subset="both",
        color_mode=color_mode,
        image_size=(img_width, img_height),
        batch_size=batch_size,
    )

    class_names = train_ds.class_names
    print(f"Classes: {class_names}")

    # Plot some images
    if mode_display:
        # Take first batch
        imgs, labels = next(iter(train_ds))
        labels_names = [class_names[i] for i in labels.numpy()]

        num_images = min(9, len(imgs))
        plot_grid_images_from_array(imgs[:num_images].numpy(), 
                                    imgs_titles=labels_names[:num_images]) 

    # Test dataset
    val_ds_len = tf.data.experimental.cardinality(val_ds)
    test_ds = val_ds.take(val_ds_len // test_split)
    val_ds = val_ds.skip(val_ds_len // test_split)

    print(f"Number of training batches: {tf.data.experimental.cardinality(train_ds)}")
    print(f"Number of validation batches: {tf.data.experimental.cardinality(val_ds)}")
    print(f"Number of test batches: {tf.data.experimental.cardinality(test_ds)}")

    # Configure dataset for performance: 
    if cache:
        train_ds = train_ds.cache()
        val_ds = val_ds.cache()    
        if shuffle:
            train_ds = train_ds.shuffle(buffer_size=1000)  

    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE) 

    # Normalize pixel values
    #normalization_layer = layers.Rescaling(1./255)
    #train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    #normalization_layer = layers.Rescaling(1./255) # to [0, 1]

    # Replicate images across channels to make them RGB
    if color_mode == 'grayscale':
        train_ds = train_ds.map(lambda x, y: (tf.image.grayscale_to_rgb(x), y))
        val_ds = val_ds.map(lambda x, y: (tf.image.grayscale_to_rgb(x), y))
        test_ds = test_ds.map(lambda x, y: (tf.image.grayscale_to_rgb(x), y))

    # Swap axes to have channels first
    #train_ds = train_ds.map(lambda x, y: (tf.transpose(x, perm=[0, 3, 1, 2]), y))

    # Take first batch
    #imgs, labels = next(iter(train_ds))
    return train_ds, val_ds, test_ds, class_names

def evaluate_model(model: tf.keras.Model, 
                   test_ds: tf.data.Dataset,
                   class_names: list):
    """Evaluate the model."""

    # Evaluate the model
    test_loss, test_acc = model.evaluate(test_ds)
    print(f"Test accuracy: {test_acc:.2f}")

    # Take first batch and predict
    imgs_test, labels_test = next(iter(test_ds))

    pred_names, pred_scores = get_predictions(model, imgs_test, class_names)
    imgs_titles = [f"True: {class_names[labels_test[i].numpy()]}, Pred: {pred_names[i]}, Score: {pred_scores[i]:.2f} " for i in range(len(pred_names))]
    plot_grid_images_from_array(imgs_test.numpy(),
                                imgs_titles=imgs_titles)
    return test_loss, test_acc

def get_callbacks(
        # Tensorboard cb
        log_dir='tb_logs',
        histogram_freq=1,
        profile_batch=None, # 2
        # Early stopping cb
        early_stopping_patience=5,
        early_stopping_monitor='val_loss',
        # Checkpoint cb
        ckpt_freq=0, # 5
        ckpt_path_file='tb_logs/ckpts/ckpt-{epoch:04d}.ckpt}',
        ckpt_monitor='val_accuracy',
        # Reduce learning rate on plateau cb
        reduce_lr_patience=5,
        reduce_lr_min=1e-6,
        reduce_lr_factor=0.2,
        reduce_lr_monitor='val_loss',
):
    """Get the callbacks."""

    tb_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir, 
                                        histogram_freq=histogram_freq,
                                        write_graph=True,
                                        write_images=True,
                                        profile_batch=profile_batch,)
    callbacks = [tb_cb]

    # Early stopping
    if early_stopping_patience > 0:
        early_stop_cb = tf.keras.callbacks.EarlyStopping(monitor=early_stopping_monitor, patience=early_stopping_patience)
        callbacks.append(early_stop_cb)

    # Checkpoint
    if ckpt_freq > 0:            
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            filepath=ckpt_path_file,
            verbose=1,
            save_weights_only=True,
            monitor=ckpt_monitor,
            save_freq=ckpt_freq,
            save_best_only=True)
        callbacks.append(checkpoint_cb)

    # Reduce learning rate on plateau
    if reduce_lr_patience > 0:
        reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(monitor=reduce_lr_monitor, 
                                                            factor=reduce_lr_factor,
                                                            patience=reduce_lr_patience, 
                                                            min_lr=reduce_lr_min)
        callbacks.append(reduce_lr_cb)
    return callbacks


def train_finetune_clf(
                        # Data
                        data_dir: str,
                        img_width: int,
                        img_height: int,
                        batch_size: int,
                        validation_split: float = 0.2,   # Percentage
                        test_split: int = 2,             # Ratio
                        color_mode: str = "grayscale",   # Images: "grayscale", "rgb" 
                        augmentation_param: dict = {'flip': 'horizontal', 'rotation': 0.2},
                        cache: bool = False,
                        shuffle: bool = True,
                        #
                        # Model
                        base_model_name: str = 'mobilenet_v2',# Pretrained model name
                        model_num_channels: int=3,       # Number of channels for the model
                        dropout: float = 0.2,            # Dropout rate
                        path_save_model: str = 'models',
                        #
                        # Train
                        initial_epochs: int = 20,        # Train the top layers for this number of epochs
                        fine_tune_at_perc: int = 0.75,         # Fine-tune from this layer onwards (in percentage)
                        base_learning_rate: float = 0.0001,
                        fine_tune_epochs: int = 10,
                        ft_learning_rate: float = 0.00001,
                        metrics: list = ['accuracy'],
                        mode_display: bool=False,
                        #
                        # TensorBoard
                        log_dir='tb_logs',
                        histogram_freq=1,
                        profile_batch=None, # 2
                        # Early stopping cb
                        early_stopping_patience=5,
                        early_stopping_monitor='val_loss',
                        # Checkpoint cb
                        ckpt_freq=0, # 5
                        ckpt_path='tb_logs/ckpts',
                        ckpt_monitor='val_accuracy',
                        # Reduce learning rate on plateau cb
                        reduce_lr_patience=5,
                        reduce_lr_min=1e-6,
                        reduce_lr_factor=0.2,
                        reduce_lr_monitor='val_loss',
    ):
    # Device name
    # tf.test.gpu_device_name()
    print(tf.__version__)
    print(f"Devices names: {device_lib.list_local_devices()}")
    # ----------------------------------------------------------------------------
    # Create data set 
    # Split in training and validation batched dataset
    train_ds, val_ds, test_ds, class_names = create_dataset(data_dir,
                                                            img_width,
                                                            img_height,
                                                            batch_size,
                                                            validation_split,
                                                            test_split,
                                                            color_mode,
                                                            cache=cache,
                                                            shuffle=shuffle,
                                                            mode_display=mode_display)
    # ----------------------------------------------------------------------------
    # Create the model
    img_shape = (img_height, img_width, model_num_channels) # Color for model trained on RGB images
    model, base_model = get_model_from_base(base_model_name,
                                img_shape,
                                len(class_names),
                                augmentation_param,
                                trainable=False,
                                dropout=dropout)

    # ----------------------------------------------------------------------------
    # Callbacks
    callbacks = get_callbacks(log_dir=log_dir, 
                              histogram_freq=histogram_freq, 
                              profile_batch=profile_batch,
                              early_stopping_patience=early_stopping_patience, 
                              early_stopping_monitor=early_stopping_monitor,
                              ckpt_freq=ckpt_freq, 
                              ckpt_path_file=ckpt_path, 
                              ckpt_monitor=ckpt_monitor,
                              reduce_lr_patience=reduce_lr_patience, 
                              reduce_lr_min=reduce_lr_min,
                              reduce_lr_factor=reduce_lr_factor, 
                              reduce_lr_monitor=reduce_lr_monitor,)

    # Save the weights using the `checkpoint_path` format
    # Contain only model's weights
    ckpt_path_file = os.path.join(ckpt_path, 'ckpt-{epoch:04d}.ckpt')
    model.save_weights(ckpt_path_file.format(epoch=0))

    # ----------------------------------------------------------------------------    
    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=metrics)

    # ----------------------------------------------------------------------------
    # Train the model: Only train the top layers
    loss0, accuracy0 = model.evaluate(val_ds)
    print(f"Initial loss: {loss0:.2f}")
    print(f"Initial accuracy: {accuracy0:.2f}")

    history = model.fit(train_ds,
                        epochs=initial_epochs,
                        validation_data=val_ds,
                        callbacks=callbacks)

    # ----------------------------------------------------------------------------
    # Plot results
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plot_loss_acc(loss, val_loss, acc, val_acc)
    ##########################################
    # Fine tuning
    base_model_num = len(base_model.layers)
    fine_tune_at = int(base_model_num*fine_tune_at_perc)
    if fine_tune_at > 0:
        base_model, model = unfreeze_model(base_model, model, fine_tune_at)

        # Compile the model
        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                        optimizer=tf.keras.optimizers.RMSprop(lr=ft_learning_rate),
                        metrics=['accuracy'])

        # Continue training the model
        total_epochs = initial_epochs + fine_tune_epochs
        history_fine = model.fit(train_ds,
                                    epochs=total_epochs,
                                    initial_epoch=history.epoch[-1],
                                    validation_data=val_ds)

        # ---------------------------------------------
        # Plot results
        acc += history_fine.history['accuracy']
        val_acc += history_fine.history['val_accuracy']

        loss += history_fine.history['loss']
        val_loss += history_fine.history['val_loss']

        plot_loss_acc(loss, val_loss, acc, val_acc)
        #plt.plot([initial_epochs-1,initial_epochs-1], plt.ylim(), label='Start Fine Tuning')

        history = history_fine

    # ---------------------------------------------
    # Evaluate the model
    test_loss, test_acc = evaluate_model(model, test_ds, class_names)
    # ---------------------------------------------
    # Save the model weights
    model.save_weights(os.path.join(path_save_model, 'final_weights'))

    # Save the entire model as a keras or legacy (SavedModel, HDF5)
    # tf.keras.models.load_model('my_model.keras')
    model.save(os.path.join(path_save_model, 'final_model.keras'))
    # ---------------------------------------------
    # Save history
    with open(os.path.join(path_save_model, 'train_history'), 'wb') as f:        
        pickle.dump(history.history, f)

    return model, history, test_loss, test_acc

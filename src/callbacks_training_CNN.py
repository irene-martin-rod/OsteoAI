###CALLBACKS
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

def create_reduce_lr_callback(monitor = "val_loss", factor = 0.5, patience = 3, min_lr = 1e-6):
    '''
    Creating an instance of ReduceLROnPlateau callback.
    
    Paramenters:
        monitor (str): Mestric to monirorize the model performance. By default: 'val_loss'.
        factor (float): Reduction tase of learning rate. By default: 0.5.
        patience (int): Epoch numbers in which learning rate has to remain before being reduce. By default: 3
        min_lr (float): Minimum learning rate. By default: 1e-6.
    
    Return:
        The instance ReduceLROnPlateau
    '''
    return ReduceLROnPlateau(
        monitor=monitor,
        factor=factor,
        patience=patience,
        min_lr=min_lr
    )

def create_early_stopping_callback(monitor = "val_loss", patience = 3, restore_best_weights=True):
    '''
    Creating an instance of EarlyStopping callback.
    
    Paramenters:
        monitor (str): Mestric to monirorize the model performance. By default: 'val_loss'.
        patience (int): Epoch numbers in which learning rate has to remain before being reduce. By default: 3
        restore_best_weights (boolean): #Reseting the weights to the best epoch's weights. By default: True
    
    Return:
        The instance EarlyStopping
    '''

    return EarlyStopping(
        monitor = monitor,      
        patience = patience,              
        restore_best_weights = restore_best_weights 
    )






def establish_checkpoints(filepath, monitor = "accuracy", verbose=1, save_best_only=True, mode='auto', save_freq='epoch'):

    '''
    Creates a ModelCheckpoint callback to save model weights during training.

    Parameters:
        filepath (str): The file path and format to save the model weights. 
                        Use placeholders like `{epoch}` or `{val_loss}` for dynamic naming.
        monitor (str): Metric to monitor. Default is "accuracy".
        verbose (int): Verbosity mode. 0 = silent, 1 = progress messages.
        save_best_only (bool): If True, only the best weights (based on monitored metric) will be saved.
        mode (str): Mode for monitoring:
                    'auto' - infers mode from the metric.
                    'min' - saves when monitored metric decreases.
                    'max' - saves when monitored metric increases.
        save_freq (str or int): Frequency of saving weights:
                                'epoch' - saves weights at the end of each epoch.
                                int - saves weights at the specified number of training steps.

    Returns:
        tensorflow.keras.callbacks.ModelCheckpoint: Configured ModelCheckpoint callback.
    '''


    checkpoint = ModelCheckpoint(
        filepath=filepath,
        monitor=monitor,
        verbose=verbose,
        save_best_only=save_best_only,
        mode=mode,
        save_freq=save_freq
    )
    return checkpoint








def train_model(model, train_dataset, steps_per_epoch = 150, epochs = 100, validation_data = None, callbacks = None, class_weight = None):
    '''
    Training a CNN model.

    Paramaters:
        model = Model that ayou want to run
        train_dataset: The training dataset object providing input data and labels.
        steps_per_epoch = number of images in each epoch
        validation_data: Validation dataset object for evaluating the model during training. If not provided, no validation will be performed.
        callbacks: callbacks to reduce learning rate and to create an early stop
        class_weight: Optional dictionary specifying weights for each class.

    Return:
        history: A History object that contains details about the training process.
    '''

    # Validate that required parameters are provided
    if train_dataset is None:
        raise ValueError("The 'train_generator' parameter is required.")
    if model is None:
        raise ValueError("The 'model' parameter is required.")
    
    # Train the model
    history = model.fit(
        train_dataset,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=validation_data,
        callbacks=callbacks,
        class_weight=class_weight
    )
    return history

    
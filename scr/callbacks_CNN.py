###CALLBACKS
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping


def create_reduce_lr_callback(monitor = "val_loss", factor = 0.5, patience = 3, min_lr = 1e-6):
    """
    Creating an instance of ReduceLROnPlateau callback.
    
    Paramenters:
        monitor (str): Mestric to monirorize the model performance. By default: 'val_loss'.
        factor (float): Reduction tase of learning rate. By default: 0.5.
        patience (int): Epoch numbers in which learning rate has to remain before being reduce. By default: 3
        min_lr (float): Minimum learning rate. By default: 1e-6.
    
    Return:
        The instance ReduceLROnPlateau
    """
    return ReduceLROnPlateau(
        monitor=monitor,
        factor=factor,
        patience=patience,
        min_lr=min_lr
    )


def create_early_stopping_callback(monitor = "val_loss", patience = 3, restore_best_weights=True):
    """
    Creating an instance of EarlyStopping callback.
    
    Paramenters:
        monitor (str): Mestric to monirorize the model performance. By default: 'val_loss'.
        patience (int): Epoch numbers in which learning rate has to remain before being reduce. By default: 3
        restore_best_weights (boolean): #Reseting the weights to the best epoch's weights. By default: True
    
    Return:
        The instance EarlyStopping
    """

    return EarlyStopping(
        monitor = monitor,      
        patience = patience,              
        restore_best_weights = restore_best_weights 
    )

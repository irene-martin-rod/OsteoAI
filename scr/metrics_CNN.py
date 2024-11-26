### CNN METRICS

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve

def plot_training_history(history):

    '''
    Plots the training and validation accuracy and loss from the history of a trained model.

    Parameters:
        history (keras.callbacks.History): The history object returned by the `fit` method of a Keras model. It contains the training and validation metrics for each epoch.

    The function creates two plots:
    1. A plot of training and validation accuracy over epochs.
    2. A plot of training and validation loss over epochs.

    '''
    # Extract training and validation metrics from the history object
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    # Define epochs (from 1 to the number of epochs + 1)
    epochs = range(1, len(train_acc) + 1)

    # Create the accuracy plot
    plt.plot(epochs, train_acc, 'b', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Display the accuracy plot
    plt.figure()

    # Create the loss plot
    plt.plot(epochs, train_loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Show the plots
    plt.show()




def plot_confusion_matrix(model, test_generator, labels=None):
    
    '''
    This function calculates the accuracy of the model on the test dataset and generates a classification report
    summarizing precision, recall, and F1-score for each class. It also computes the confusion matrix and displays
    it as a heatmap for visual analysis. The labels for the confusion matrix can be passed explicitly or derived
    automatically from the test generator.

    Parameters:
    -----------
    model : keras.Model
        The trained model to be evaluated.
    test_generator : keras.preprocessing.image.ImageDataGenerator
        Data generator that provides test images and their corresponding true labels.
    labels : list, optional
        List of class labels for the confusion matrix. If not provided, uses the class labels from the generator.

    Returns:
    --------
    accuracy : float
        Accuracy of the model on the test dataset.
    class_report : str
        Classification report including precision, recall, F1-score, and support for each class.
    cm : numpy.ndarray
        Confusion matrix showing the true vs. predicted labels.
        '''


    # Use class labels from the generator if none are provided
    if labels is None:
        labels = list(test_generator.class_indices.keys())

    # Get predictions from the model
    test_predictions = model.predict(test_generator, steps=len(test_generator), verbose=1)
    
    # Convert predicted probabilities to class labels
    test_pred_labels = np.argmax(test_predictions, axis=1)
    
    # Get true labels from the test generator
    test_true_labels = test_generator.classes
    
    # Calculate accuracy
    accuracy = accuracy_score(test_true_labels, test_pred_labels)
    print(f"Accuracy on test dataset: {accuracy:.4f}")
    
    # Print classification report
    class_report = classification_report(test_true_labels, test_pred_labels, target_names=labels)
    print("\nClassification Report:\n", class_report)
    
    # Compute confusion matrix
    cm = confusion_matrix(test_true_labels, test_pred_labels)
    
    # Display the confusion matrix as a heatmap
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix for Test Data')
    plt.tight_layout()
    plt.show()
    
    return accuracy, class_report, cm



def plot_auc_curve(test_predictions, test_true_labels):

    '''
    This function generates and displays the AUC-ROC curve for a binary classification model.
    It computes the Area Under the Curve (AUC) and plots the ROC curve, displaying the AUC score on the graph.
    
    Parameters:
    - test_predictions: A list or array containing the predicted probabilities from the model for the positive class 
                         (usually values between 0 and 1).
    - test_true_labels: A list or array containing the true labels (actual ground truth) for the test data.
                        For binary classification, these should be either 0 or 1.
    
    Process:
    1. Calculates the AUC score using `roc_auc_score` to evaluate the model's ability to distinguish between positive and negative classes.
    2. Computes the ROC curve using `roc_curve`, which plots the true positive rate (TPR) against the false positive rate (FPR) at various thresholds.
    3. Plots the ROC curve and shows the AUC value on the graph.
    4. Prints the AUC value to the console for reference.
    
    Return:
    - AUC value
    '''
    
    # Calculate the AUC score
    auc = roc_auc_score(test_true_labels, test_predictions)
    print(f"AUC: {auc}")
    
    # Calculate the ROC curve (False Positive Rate and True Positive Rate)
    fpr, tpr, thresholds = roc_curve(test_true_labels, test_predictions)
    
    # Plot the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}', color='blue')  # ROC curve plot
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')  # Reference line for random classifier
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.show()

    return auc



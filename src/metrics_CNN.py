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




def plot_confusion_matrix(model, test_dataset, class_names):
    """
    Generate and display a confusion matrix for a model using a tf.data.Dataset.

    Parameters:
        model : keras.Model
            Trained model to evaluate.
        test_dataset : tf.data.Dataset
            Dataset containing test images and labels.
        class_names : list of str
            List of class names for the dataset.

    Returns:
        accuracy : float
            Accuracy of the model on the test dataset.
        class_report : str
            Detailed classification report.
        cm : numpy.ndarray
            Confusion matrix.
    """
    # Extract all test images and labels
    true_labels = []
    for _, labels in test_dataset:
        true_labels.extend(labels.numpy())
    true_labels = np.array(true_labels)

    # Get predictions from the model
    test_predictions = model.predict(test_dataset, verbose=1)
    
    # Convert predicted probabilities to class labels
    if test_predictions.shape[-1] > 1:  # Multi-class case
        pred_labels = np.argmax(test_predictions, axis=1)
    else:  # Binary classification case
        pred_labels = (test_predictions > 0.5).astype(int).flatten()

    # Calculate accuracy
    accuracy = accuracy_score(true_labels, pred_labels)
    print(f"Accuracy on test dataset: {accuracy:.4f}")

    # Print classification report
    class_report = classification_report(true_labels, pred_labels, target_names=class_names)
    print("\nClassification Report:\n", class_report)

    # Compute confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    
    # Display the confusion matrix as a heatmap
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix for Test Data')
    plt.tight_layout()
    plt.show()

    return accuracy, class_report, cm



def plot_auc_curve(model, test_dataset):
    """
    Generates and displays the AUC-ROC curve for a binary classification model using tf.data.Dataset.

    Parameters:
        model : keras.Model
            Trained model to evaluate.
        test_dataset : tf.data.Dataset
            Dataset containing test images and labels.

    Process:
        1. Extract true labels from `test_dataset`.
        2. Predict probabilities for test images using the model.
        3. Calculate AUC and the ROC curve.
        4. Plot the ROC curve, including a reference line and the AUC score.

    Returns:
        auc : float
            Area Under the Curve (AUC) value.
    """
    # Extract true labels from the dataset
    true_labels = []
    for _, labels in test_dataset:
        true_labels.extend(labels.numpy())
    true_labels = np.array(true_labels)

    # Get predicted probabilities from the model
    predictions = model.predict(test_dataset, verbose=1).flatten()

    # Calculate the AUC score
    auc = roc_auc_score(true_labels, predictions)
    print(f"AUC: {auc:.4f}")

    # Calculate the ROC curve
    fpr, tpr, thresholds = roc_curve(true_labels, predictions)

    # Plot the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}', color='blue')  # ROC curve plot
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess', color='red')  # Reference line for random classifier
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=15)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    return auc



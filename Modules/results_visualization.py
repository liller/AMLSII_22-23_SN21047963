import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd


def plot_history(history,path):
    """
    plot accuracy and loss learning curves
    Args:
        history: the results of training process, value of fit_generator
        path: path to store plot
    """
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(history.history['loss'], color='b', label="Training loss")
    ax[0].plot(history.history['val_loss'], color='r', label="Validation loss", axes=ax[0])
    legend = ax[0].legend(loc='best', shadow=True)
    ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")
    ax[1].plot(history.history['val_accuracy'], color='r', label="Validation accuracy")
    legend = ax[1].legend(loc='best', shadow=True)
    if os.path.isfile(path):
        plt.savefig(path)


def plot_confusion_matrix(y_test_array,y_predict):
    """
    plot confusion matrix on test set
    Args:
        y_test_array: the real label of test set
        y_predict: the prediction of rest set
    """
    disp = confusion_matrix(y_test_array, y_predict, normalize='true')
    plt.figure(figsize=(5, 5))
    sns.heatmap(disp,annot=True,cmap='Blues')
    plt.ylim(0, 5)
    plt.title('Confusion matrix')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.show()





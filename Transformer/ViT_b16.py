"""Vit_b16 model contains the content about model construction, training and testing.

"""

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from Modules.results_visualization import plot_confusion_matrix, plot_history

import numpy as np

np.random.seed(2)


from sklearn.metrics import f1_score
import keras
from keras.optimizers import Adam
from vit_keras import vit
from keras.callbacks import ReduceLROnPlateau




class Vit_b16:
    """
     Define class of Vit_b16 and compile

    """
    def __init__(self):
        print("===== Construct Vit_b16 model =====")
        vit_model = vit.vit_b16(
            image_size=(224, 224),
            activation='softmax',
            pretrained=True,
            include_top=True,  ##是否保留fc
            pretrained_top=False,
            classes=5,  ##更改fc的classes数量
            weights='imagenet21k')
        self.model = keras.Sequential([
            vit_model
        ],name = 'vision_transformer')
        print("Summary of the Vit_b16 model")
        self.model.summary()
        optimizer = Adam(lr=3e-6)
        self.model.model.compile(optimizer = optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    def train(self, train_batch, valid_batch, path, epochs= 30, plot= True):
        """
        training model and plot the learning curves
        Args:
            train_batch: training set
            valid_batch: validation set
            path: path to save the learning curves
        Returns:
            result of training process contains accuracy and loss
        """
        print("===== Training Vit_b16 model =====")
        learning_rate_reduction = ReduceLROnPlateau(monitor="val_loss",
                                                     patience=3,
                                                     verbose=1,
                                                     factor=0.5,
                                                     min_lr=0.0000001)
        history = self.model.fit_generator(train_batch, steps_per_epoch=len(train_batch), validation_data = valid_batch,
                                  validation_steps=len(valid_batch), epochs=epochs, callbacks=[learning_rate_reduction], verbose=1)
        if plot:
            plot_history(history, path)
        return history

    def train_weights(self, train_batch, valid_batch, class_weights, path, epochs= 30, plot= True):
        """
        training model with class_weights and plot the learning curves
        Args:
            train_batch: training set
            valid_batch: validation set
            path: path to save the learning curves
        Returns:
            result of training process contains accuracy and loss
        """
        print("===== Training Vit_b16 model =====")
        learning_rate_reduction = ReduceLROnPlateau(monitor="val_loss",
                                                     patience=3,
                                                     verbose=1,
                                                     factor=0.5,
                                                     min_lr=0.00001)

        history = self.model.fit_generator(train_batch, steps_per_epoch=len(train_batch), validation_data = valid_batch,
                                  validation_steps=len(valid_batch), epochs=epochs, callbacks=[learning_rate_reduction],
                                            verbose=1, class_weight= class_weights)
        if plot:
            plot_history(history, path)
        return history


    def test(self, test_batch, confusion_mat=False):
        """
        verify the model on test set
        Args:
            test_batch: test data set
        Returns:
            marco weighted F1-score
        """
        print("===== Test Vit_b16 model on test set=====")
        pred=self.model.predict_generator(test_batch, steps = len(test_batch), verbose=1)
        pred = np.round(pred)
        predicted_labels = np.array(np.argmax(pred, axis=1))
        true_labels = np.array(test_batch.classes)
        score = f1_score(true_labels, predicted_labels, average='macro')
        if confusion_mat:
            plot_confusion_matrix(np.argmax(test_batch,axis = 1),np.argmax(pred,axis = 1))
        return score



"""VGG_finetune model contains the content about model construction, training and testing.

"""
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
np.random.seed(2)
from sklearn.metrics import f1_score
from keras.models import Sequential
from keras.layers import Dense,GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.applications.vgg19 import VGG19
from keras.callbacks import ReduceLROnPlateau
from Modules.results_visualization import plot_confusion_matrix, plot_history




class VGG_finetune:
    """
     Define class of VGG_finetune and compile

    """
    def __init__(self):
        print("===== Construct VGG_finetune model =====")
        base_model = VGG19(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
        self.model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(256, activation="relu"),
            Dense(5, activation='softmax')
        ])
        print("===== Summary of the VGG_finetune model")
        self.model.summary()
        for layer in base_model.layers[12:]:
            layer.trainable = True
        for layer in base_model.layers[0:12]:
            layer.trainable = False
        self.model.summary()
        optimizer = Adam(lr=2e-5)
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
        print("===== Training VGG_finetune model =====")
        learning_rate_reduction = ReduceLROnPlateau(monitor="val_loss",
                                                     patience=3,
                                                     verbose=1,
                                                     factor=0.5,
                                                     min_lr=0.00001)
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
        print("===== Training VGG_finetune model =====")
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
        print("===== Test VGG_finetune model on test set=====")
        pred=self.model.predict_generator(test_batch, steps = len(test_batch), verbose=1)
        pred = np.round(pred)
        predicted_labels = np.array(np.argmax(pred, axis=1))
        true_labels = np.array(test_batch.classes)
        score = f1_score(true_labels, predicted_labels, average='macro')
        if confusion_mat:
            plot_confusion_matrix(np.argmax(test_batch,axis = 1),np.argmax(pred,axis = 1))
        return score
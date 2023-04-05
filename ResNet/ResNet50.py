"""ResNet model contains the content about model construction, training and testing.

"""

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from Modules.results_visualization import plot_confusion_matrix, plot_history
import numpy as np
np.random.seed(2)
from sklearn.metrics import f1_score
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.applications.resnet50 import ResNet50
from keras.callbacks import ReduceLROnPlateau




class ResNet:
    """
     Define class of ResNet and compile

    """
    def __init__(self):
        print("===== Construct ResNet model =====")
        base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224,224,3))
        self.model = Sequential([
            base_model,
            Dropout(0.5),
            GlobalAveragePooling2D(),
            Dense(1024, activation="relu"),
            Dropout(0.25),
            Dense(5, activation='softmax')
        ])
        print("Summary of the ResNet model")
        self.model.summary()
        optimizer = Adam(lr=3e-5)
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
        print("===== Training ResNet model =====")
        rng = [i for i in range(30)]
        y = [self.lr_schedule(x) for x in rng]
        print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(y[0], max(y), y[-1]))
        lr_callback = tf.keras.callbacks.LearningRateScheduler(self.lr_schedule, verbose=True)
        history = self.model.fit_generator(train_batch, steps_per_epoch=len(train_batch), validation_data = valid_batch,
                                  validation_steps=len(valid_batch), epochs=epochs, callbacks=[lr_callback], verbose=1)
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
        print("===== Training ResNet model =====")
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
        print("===== Test ResNet model on test set=====")
        pred=self.model.predict_generator(test_batch, steps = len(test_batch), verbose=1)
        pred = np.round(pred)
        predicted_labels = np.array(np.argmax(pred, axis=1))
        true_labels = np.array(test_batch.classes)
        score = f1_score(true_labels, predicted_labels, average='macro')
        if confusion_mat:
            plot_confusion_matrix(np.argmax(test_batch,axis = 1),np.argmax(pred,axis = 1))
        return score

    def lr_schedule(self, epoch):
        """
        control lr change
        Args:
            epoch: num of epoch
        Returns:
            lr during the amount of epochs
        """
        LR_START = 0.0005
        LR_MAX = 0.001
        LR_MIN = 0.00001
        LR_RAMPUP_EPOCHS = 0
        LR_SUSTAIN_EPOCHS = 0
        LR_EXP_DECAY = .8
        if epoch < LR_RAMPUP_EPOCHS:
            lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
        elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
            lr = LR_MAX
        else:
            lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY ** (epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN
        return lr


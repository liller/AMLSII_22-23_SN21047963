import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from Modules.results_visualization import plot_confusion_matrix, plot_history
import numpy as np
np.random.seed(2)
from sklearn.metrics import f1_score
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.callbacks import ReduceLROnPlateau




class Base_CNN:
    def __init__(self):
        # Set the CNN model
        print("Construct Base_CNN model =====")
        self.model = Sequential([
            Conv2D(filters=32, kernel_size=(5, 5), padding='Same',activation='relu', input_shape=(224, 224, 3)),
            Conv2D(filters=32, kernel_size=(5, 5), padding='Same',activation='relu'),
            MaxPool2D(pool_size=(2, 2)),

            Conv2D(filters=64, kernel_size=(3, 3), padding='Same',activation='relu'),
            Conv2D(filters=64, kernel_size=(3, 3), padding='Same',activation='relu'),
            MaxPool2D(pool_size=(2, 2), strides=(2, 2)),

            Flatten(),
            Dense(256, activation="relu"),
            Dense(5, activation='softmax')
        ])
        print("Summary of the Base_CNN model")
        self.model.summary()
        optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
        self.model.model.compile(optimizer = optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    def train(self, train_batch, valid_batch, path, epochs= 30, plot= True):
         print("Training CNN model =====")
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


    def test(self, test_batch, confusion_mat=False):
        print("Test CNN model on test set=====")
        pred=self.model.predict_generator(test_batch, steps = len(test_batch), verbose=1)
        pred = np.round(pred)
        predicted_labels = np.array(np.argmax(pred, axis=1))
        true_labels = np.array(test_batch.classes)
        score = f1_score(true_labels, predicted_labels, average='macro')
        if confusion_mat:
            plot_confusion_matrix(np.argmax(test_batch,axis = 1),np.argmax(pred,axis = 1))
        return score
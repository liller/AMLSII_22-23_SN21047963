import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from Modules.results_visualization import plot_confusion_matrix, plot_history
import numpy as np
np.random.seed(2)
from sklearn.metrics import f1_score
from keras.models import Sequential
from keras.layers import Input, Dense, GlobalAveragePooling2D
from keras.applications.vgg19 import VGG19



class VGG_feature_extractor:
    def __init__(self):
        # Set the CNN model
        print("Construct Base_CNN model =====")
        base_model = VGG19(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
        self.model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(5, activation='softmax')
        ])
        print("Summary of the Base_CNN model")
        self.model.summary()
        # for layer in self.model.layers[0:21]:
        #     layer.trainable = False
        # optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
        # self.model.model.compile(optimizer = optimizer, loss="categorical_crossentropy", metrics=["accuracy"])


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
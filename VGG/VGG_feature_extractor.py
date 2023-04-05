"""VGG_feature_extractor model contains the content about model construction and testing.

"""
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
    """
     Define class of VGG_feature_extractor and compile

    """
    def __init__(self):
        print("===== Construct VGG_feature_extractor model =====")
        base_model = VGG19(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
        self.model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(5, activation='softmax')
        ])
        print("Summary of the VGG_feature_extractor model")
        self.model.summary()


    def test(self, test_batch, confusion_mat=False):
        """
        verify the model on test set
        Args:
            test_batch: test data set
        Returns:
            marco weighted F1-score
        """
        print("===== Test CNN VGG_feature_extractor on test set=====")
        pred=self.model.predict_generator(test_batch, steps = len(test_batch), verbose=1)
        pred = np.round(pred)
        predicted_labels = np.array(np.argmax(pred, axis=1))
        true_labels = np.array(test_batch.classes)
        score = f1_score(true_labels, predicted_labels, average='macro')
        if confusion_mat:
            plot_confusion_matrix(np.argmax(test_batch,axis = 1),np.argmax(pred,axis = 1))
        return score
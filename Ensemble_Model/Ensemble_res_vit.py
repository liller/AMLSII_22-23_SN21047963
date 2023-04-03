import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
np.random.seed(2)
from ResNet.ResNet50 import *
from Transformer.ViT_b16 import *




class Ensemble_model:
    def __init__(self):
        # Set the CNN model
        print("Construct Base_CNN model =====")
        self.resnet_model = ResNet()
        self.resnet_model.load_weights('./model/Pre-trained-resnet_aug.h5')
        self.vit_model = Vit_b16()
        self.vit_model.load_weights('./model/Pre-trained-transformer_v16_aug.h5')


    def test(self, test_batch, confusion_mat=False):
        print("Test CNN model on test set=====")
        resnet_pred = self.resnet_model.predict_generator(test_batch, steps=len(test_batch), verbose=1)
        vit_pred = self.vit_model.predict_generator(test_batch, steps=len(test_batch), verbose=1)
        total_pred = 0.5 * resnet_pred + 0.5 * vit_pred
        pred = np.round(total_pred)
        predicted_labels = np.array(np.argmax(total_pred, axis=1))
        true_labels = np.array(test_batch.classes)
        score = f1_score(true_labels, predicted_labels, average='macro')
        if confusion_mat:
            plot_confusion_matrix(np.argmax(test_batch,axis = 1),np.argmax(pred,axis = 1))
        return score




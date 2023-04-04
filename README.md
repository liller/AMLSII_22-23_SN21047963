# AMLSII__22-23_SN21047963
## Intro
This report discusses the effectiveness of using multiple Convolutional Neural Networks (CNNs) and Transformer architectures for image multi-classification tasks in cassava leaf disease recognition. I conducted experiments on the cassava leaf disease dataset using basic CNN networks, VGG, ResNet, and Vision Transformer models. To enhance the model's robustness and generalization ability, I also introduced additional techniques such as Batch Normaliza-tion, Dropout, Data Augmentation, and Class Weight. The experimental results show that, with the reasonable use of transfer learning (fine-tuning) techniques, the use of the more complex ResNet model performs better than the basic CNN model on the test set, while also demonstrating the excellent performance and potential of the Vision Trans-former architecture in the field of image vision. Additional-ly, the data augmentation, dropout, and BN techniques used in the experiment can effectively alleviate the overfitting and gradient vanishing phenomena that may occur.


The final accuracy results is shown in the following table:
|  | Task A1 | Task A2 | Task B1 | Task B2 |
| :----:| :----:| :----: |:----: |:----: |
| CNN1 | 91% | 81% | 100% | 82% |
| CNN1(VGG) | / | / | / | 70.2% |
| SVM | 88% | 83% | / | / |
| SVM(&detector) | 98% | 72% | 60.7% | 71% |
| SVM(&landmarks) | / | 90% | / | / |

## Role of each file
The current project structure is shown below
```
.
├── Base_Model
│   ├── Base_CNN.py
│   └── Base_CNN_increased_layers.py
├── Dataset
│   ├── test_labels.csv
│   ├── train.csv
│   └── train_labels.csv
├── Ensemble_Model
│   └── Ensemble_res_vit.py
├── Model
├── Modules
│   ├── pre_processing.py
│   └── results_visualization.py
├── ResNet
│   └── ResNet50.py
├── Results_img
├── Transformer
│   ├── ViT_b16.py
│   └── ViT_b32.py
├── VGG
│   ├── VGG_feature_extractor.py
│   ├── VGG_finetune.py
│   └── VGG_training_from_scratch.py
└── main.py

```

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
**main**: Contains all the core functions that will be executed sequentially for data loading, pre-processing, splitting dataset, data augmentation, model instance creation, model training, result prediction and evaluations.

. 
**Modules**: Contains two files including pre_processing and results_visualization. 
**Base_Model**: Contains  2 block basic CNN model and 3 block basic CNN model. 
**VGG**: Contains 3 models trained by different approaches in transfer learning field, including training as a feature extractor. training by fine-tuning and training from scratch. 
**ResNet** 
**Transformer**: Contains ViT_b16 and ViT_b32 pretrained model. 
**Ensemble_Model**: Contains the ensemble model of ResNet50 and ViT_b16. 
**environment.yml**: Contain all the dependencies this project need. 

## Requirements

## How to start
### 1. Setup
1. Create a new virtual conda environment based on the provided environment.yml file and execute the following statement in the project path. 

```
conda env create -f environment.yml
```
2. Activate this conda virtual environment. 
```
conda activate cassava_leaf_disease

```

### 2. Prepare Dataset
The dataset could be downloaded from kaggle or run the commands.
```
kaggle competitions download -c cassava-leaf-disease-classification
```
After downloading the datset, place the "train_images" folder in the "Dataset" directory mentioned above。

### 3. Run code

If all the dependencies required for the current project are already installed and placed the dataset in the specific location, you can run main.py 
```
python main.py
```
The program will read the images from the Datasets directory and automatically create a new folder to store the test set, and then start pre-processing, model building, model training, prediction and evaluation process. 

**Note**: when you copy the datasets to the Datasets directory, you only need to copy the "train_images" folder. The program will automatically divide the test set from the above datasets and create a new directory to store the test data. The ratio of training set, validation set and test set is 8:1:1. 

Due to the monthly limitation of Git LFS uploading large files, it is not possible to upload the trained model files to github, so it will take some time to train all the model. Specific hyperparameters can be found at the end instruction.



## CNN Model paramater

| model | input size  |  optimizer | learning rate | batch_size | epochs |
| :----:| :----:| :----: |:----: |:----: | :----: |
| CNN1(Task A) | (218,178,3) | RMSprop | 0.001 | 32 | 30 | 
| CNN1(Task B) | (500,500,3) | RMSprop | 0.001 with ReduceLROnPlateau callback| 32 | 30 | 
| CNN2 |（224，224，3）  |  Adam|  0.01 with ReduceLROnPlateau callbacks| 32 | 30 | 

## SVM Model paramater 
| model | input size  |  kernal | 
| :----:| :----:| :----: |
| SVM(Task A & B)  | (5000, 38804) | linear | 
| SVM(Task A ) with detector | (4868, 128) |linear  |
| SVM(Task B) with detector | (7984, 128) |poly  | 
| SVM(Task A2) with landmarks  | (4833, 136) | linear |  




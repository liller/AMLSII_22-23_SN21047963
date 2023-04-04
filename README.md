# AMLSII__22-23_SN21047963
## Intro
This report discusses the effectiveness of using multiple Convolutional Neural Networks (CNNs) and Transformer architectures for image multi-classification tasks in cassava leaf disease recognition. I conducted experiments on the cassava leaf disease dataset using basic CNN networks, VGG, ResNet, and Vision Transformer models. To enhance the model's robustness and generalization ability, I also introduced additional techniques such as Batch Normaliza-tion, Dropout, Data Augmentation, and Class Weight. The experimental results show that, with the reasonable use of transfer learning (fine-tuning) techniques, the use of the more complex ResNet model performs better than the basic CNN model on the test set, while also demonstrating the excellent performance and potential of the Vision Trans-former architecture in the field of image vision. Additional-ly, the data augmentation, dropout, and BN techniques used in the experiment can effectively alleviate the overfitting and gradient vanishing phenomena that may occur.


The final accuracy results is shown in the following table:
|  | Base CNN | VGG | ResNet | ViT | Ensemble |
| :----:| :----:| :----: |:----: |:----: |:----: |
| Marco weighted F1-Score | 66% | 79% | 84% | 81% | 84% |
| Marco weighted F1-Score | 45% | 64% | 70% | 69% | 73% |
Specific hyperparameters and techniques can be found at the end of instruction.

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

**Modules**: Contains two files including pre_processing and results_visualization. Pre_processing do performing image processing such as normalization on the image data, divide the training set, validation set and test set. Results_visualization do plotting accuracy results and loss value curves, plotting prediction result confusion matrix.

**Base_Model**: Contains  2 block basic CNN model and 3 block basic CNN model. 

**VGG**: Contains 3 models trained by different approaches in transfer learning field, including training as a feature extractor. training by fine-tuning and training from scratch. 

**ResNet**: Contains model of ResNet50

**Transformer**: Contains ViT_b16 and ViT_b32 pretrained model. 

**Ensemble_Model**: Contains the ensemble model of ResNet50 and ViT_b16. 

**environment.yml**: Contain all the dependencies this project need. 


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
You could also specify the graphics card on which to execute the program with the following command
```
CUDA_VISIBLE_DEVICES=" " python main.py 
```
Or if you like using the Notebook to run the code, you could also do the following command before import external library.
```
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ' '
```

The program will read the images from the Datasets directory and automatically create a new folder to store the test set, and then start pre-processing, model building, model training, prediction and evaluation process. 

**Note**: when you copy the datasets to the Datasets directory, you only need to copy the "train_images" folder. The program will automatically divide the test set from the above datasets and create a new directory to store the test data. The ratio of training set, validation set and test set is 8:1:1. 

Due to the monthly limitation of Git LFS uploading large files, it is not possible to upload the trained model files to github, so it will take some time to train all the model. 



## Base Model 

| Model Description | Input Size | Optimiser |  LR  |  Class Weight |  Data Augmentation | Batch Size  | Epochs |
| :----:| :----:| :----: |:----: |:----: | :----: | :----: | :----: |
| Base model | (224,224,3) | RMSprop | 0.001 | 32 | 30 |  30 |  30 | 
| Add more layers and droupout | (500,500,3) | RMSprop | 0.001 with ReduceLROnPlateau callback| 32 | 30 |  30 |  30 | 


## SVM Model paramater 
| model | input size  |  kernal | 
| :----:| :----:| :----: |
| SVM(Task A & B)  | (5000, 38804) | linear | 
| SVM(Task A ) with detector | (4868, 128) |linear  |
| SVM(Task B) with detector | (7984, 128) |poly  | 
| SVM(Task A2) with landmarks  | (4833, 136) | linear |  




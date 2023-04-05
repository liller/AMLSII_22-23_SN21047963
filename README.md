# AMLSII__22-23_SN21047963
## Description
This report discusses the effectiveness of using multiple Convolutional Neural Networks (CNNs) and Transformer architectures for image multi-classification tasks in cassava leaf disease recognition. I conducted experiments on the cassava leaf disease dataset using basic CNN networks, VGG, ResNet, and Vision Transformer models. To enhance the model's robustness and generalization ability, I also introduced additional techniques such as Batch Normalization, Dropout, Data Augmentation, and Class Weight. The experimental results show that, with the reasonable use of transfer learning (fine-tuning) techniques, the use of the more complex ResNet model performs better than the basic CNN model on the test set, while also demonstrating the excellent performance and potential of the Vision Transformer architecture in the field of image vision. Additionally, the data augmentation, dropout, and BN techniques used in the experiment can effectively alleviate the overfitting and gradient vanishing phenomena that may occur.


The final accuracy results is shown in the following table:
|  | Base CNN | VGG | ResNet | ViT | Ensemble |
| :----:| :----:| :----: |:----: |:----: |:----: |
| Marco weighted F1-Score | 66% | 79% | 84% | 81% | 84% |
| Marco avg F1-Score | 45% | 64% | 70% | 69% | 73% |

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
**main.py**: Contains all the core functions that will be executed sequentially for data loading, pre-processing, splitting dataset, data augmentation, model instance creation, model training, result prediction and evaluations. 

**Modules**: Contains two files including pre_processing and results_visualization. Pre_processing do performing image processing such as normalization on the image data, divide the training set, validation set and test set. Results_visualization do plotting accuracy results and loss value curves, plotting prediction result confusion matrix.

**Base_Model**: Contains  2 block basic CNN model and 3 block basic CNN model. 

**VGG**: Contains 3 models trained by different approaches in transfer learning field, including training as a feature extractor. training by fine-tuning and training from scratch. 

**ResNet**: Contains model of ResNet50

**Transformer**: Contains ViT_b16 and ViT_b32 pretrained model. 

**Ensemble_Model**: Contains the ensemble model of ResNet50 and ViT_b16. 

**environment.yml**: Contain all the dependencies this project need. 


## Getting started
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


## Model Performance & Hyperparamater

### Base Model 
<div align=center>
<img src="https://github.com/liller/AMLSII__22-23_SN21047963/blob/master/Model_architecture/Base_model_3Block.jpg" width="814" height="300">
</div>

| Model Description | Input Size | Optimiser |  LR  |  Class_weight |  Data_Aug | Batch Size  | Epochs | Marco weighted F1-Score | Marco avg F1-Score |
| :----:| :----:| :----: |:----: |:----: | :----: | :----: | :----: | :----: | :----: |
| Base model | (800,600,3) | RMSprop | 0.001 | / | / | 64 |  30 | 65% | 34% |
| Add more layers and droupout | (800,600,3) | RMSprop | 0.001 with ReduceLROnPlateau | / | / | 64 |  30 |  66% | 45% |


### VGG Model 
<div align=center>
<img src="https://github.com/liller/AMLSII__22-23_SN21047963/blob/master/Model_architecture/VGG.jpg">
</div>

| Model Description | Input Size | Optimiser |  LR  |  Class_weight |  Data_Aug | Batch Size  | Epochs | Marco weighted F1-Score | Marco avg F1-Score |
| :----:| :----:| :----: |:----: |:----: | :----: | :----: | :----: | :----: | :----: |
| Feature extractor | (224,224,3) | / | / | / | / | / |  / | 71% | 19% |
| Training from scratch | (224,224,3) | Adam | 2e-5 with ReduceLROnPlateau | / | / | 256 |  30 |  74% | 56% |
| Fine-tune by freezing 2 block | (224,224,3) | Adam | 2e-5 with ReduceLROnPlateau | / | / | 256 |  30 |  79% | 64% |
| Fine-tune by freezing 2 block | (224,224,3) | Adam | 3e-6 with ReduceLROnPlateau | True | / | 256 |  30 |  67% | 55% |


### ResNet Model 


| Model Description | Input Size | Optimiser |  LR  |  Class_weight |  Data_Aug | Batch Size  | Epochs | Marco weighted F1-Score | Marco avg F1-Score |
| :----:| :----:| :----: |:----: |:----: | :----: | :----: | :----: | :----: | :----: |
| ResNet50 | (224,224,3) | Adam | 3e-6 with ReduceLROnPlateau | / | / | 256 |  30 |  75% | 58% |
| ResNet50 | (224,224,3) | Adam | [1.15e-5, 0.001] with lr schedule | / | / | 256 |  30 |  83% | 69% |
| ResNet50 | (224,224,3) | Adam | [1.15e-5, 0.001] with lr schedule | / | True | 256 |  30 |  84% | 70% |
| ResNet50 (Fine-tune) | (224,224,3) | Adam | 3e-6 with ReduceLROnPlateau | / | / | 256 |  30 |  74% | 56% |



### Vision Transformer Model 

| Model Description | Input Size | Optimiser |  LR  |  Class_weight |  Data_Aug | Batch Size  | Epochs | Marco weighted F1-Score | Marco avg F1-Score |
| :----:| :----:| :----: |:----: |:----: | :----: | :----: | :----: | :----: | :----: |
| Vit_b32 | (224,224,3) | Adam | 3e-6 with ReduceLROnPlateau | / | / | 256 |  30 |  77% | 62% |
| Vit_b32 | (224,224,3) | Adam | 3e-6 with ReduceLROnPlateau | / | True | 256 |  50 |  79% | 65% |
| Vit_b32 | (224,224,3) | Adam | [3.03e-7, 1e-5] with lr schedule | True | / | 256 |  50 |  66% | 56% |
| Vit_b16 | (224,224,3) | Adam | 3e-6 with ReduceLROnPlateau | / | True | 256 |  30 |  81% | 69% |



### Ensemble Model 

| Model Description | Input Size | Optimiser |  LR  |  Class_weight |  Data_Aug | Batch Size  | Epochs | Marco weighted F1-Score | Marco avg F1-Score |
| :----:| :----:| :----: |:----: |:----: | :----: | :----: | :----: | :----: | :----: |
| Reset50 & ViT_b16 | (224,224,3) | Adam | 3e-6 with ReduceLROnPlateau | / | True | 256 |  50 |  84% | 73% |






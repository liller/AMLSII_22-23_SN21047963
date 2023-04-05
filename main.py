"""Contains all the core functions that will be executed sequentially for data loading, pre-processing,
splitting dataset, data augmentation, model instance creation, model training, result prediction and evaluations.

"""
from Modules.pre_processing import loading_and_splitting_data, data_generator, data_augmentation
from Base_Model.Base_CNN import *
from Base_Model.Base_CNN_increased_layers import *
from VGG.VGG_feature_extractor import *
from VGG.VGG_finetune import *
from VGG.VGG_training_from_scratch import *
from ResNet.ResNet import *
from Transformer.ViT_b32 import *
from Transformer.ViT_b16 import *
from Ensemble_Model.Ensemble_res_vit import *



dataset = './Dataset/train_images/'
dataset_test = './Dataset/test_images/'
train_labels_path = './Dataset/train_labels.csv'
test_labels_path = './Dataset/test_labels.csv'

#=============================Loading, Splitting dataset =============================
train_labels, test_labels = loading_and_splitting_data(dataset,dataset_test)
train_batch, valid_batch, test_batch, class_weights = data_generator(train_labels_path, test_labels_path)
train_aug_batch, valid_aug_batch, test_aug_batch, class_weights = data_augmentation(train_labels_path, test_labels_path)


#============================= Base Model =============================
#============== CNN With 2 Block ==============
model = Base_CNN()
path_results = './Results_img/Base_CNN_2Block.png'
model.train(train_batch,valid_batch,path=path_results)
acc_Base_CNN_score = model.test(test_batch)
print(f'The marco F1 score of Base_CNN_2Block model is: {acc_Base_CNN_score}')

#============== CNN With 3 Block ==============
model = Base_CNN_increased_layer()
path_results = './Results_img/Base_CNN_3Block.png'
model.train(train_batch,valid_batch,path=path_results)
acc_Base_CNN_increased_layer_score = model.test(test_batch)
print(f'The marco F1 score of Base_CNN_3Block model is: {acc_Base_CNN_increased_layer_score}')



#============================= VGG Model =============================
#============== Feature_Extractor ==============
model = VGG_feature_extractor()
# model.train(train_batch,valid_batch,path=path_results)
acc_VGG_feature_extractor_score = model.test(test_batch)
print(f'The marco F1 score of VGG_feature_extractor model is: {acc_VGG_feature_extractor_score}')


#============== Fine-tune By Freezing last 2 Block ==============
model = VGG_finetune()
path_results = './Results_img/VGG_fine_tune.png'
model.train(train_batch,valid_batch,path=path_results)
acc_VGG_fine_tune_score = model.test(test_batch)
print(f'The marco F1 score of VGG_fine_tune model is: {acc_VGG_fine_tune_score}')

#============== Fine-tune By Freezing last 2 Block with class_weight ==============
model = VGG_finetune()
path_results = './Results_img/VGG_Fine_tune_weighted.png'
model.train_weights(train_batch,valid_batch, class_weights, path=path_results )
acc_VGG_fine_tune_weighted_score = model.test(test_batch)
print(f'The marco F1 score of VGG_Fine_tune_weighted model is: {acc_VGG_fine_tune_weighted_score}')

#============== Training From Scratch ==============
model = VGG_scratch()
path_results = './Results_img/VGG_scratch.png'
model.train(train_batch,valid_batch,path=path_results)
acc_VGG_fine_tune_score = model.test(test_batch)
print(f'The marco F1 score of VGG_scratch model is: {acc_VGG_fine_tune_score}')


#============================= ResNet Model =============================
#============== ResNet50 ==============
model = ResNet()
path_results = './Results_img/ResNet50.png'
model.train(train_batch,valid_batch,path=path_results)
acc_ResNet_lrfn_score = model.test(test_batch)
print(f'The marco F1 score of ResNet50 model is: {acc_ResNet_lrfn_score}')


#============== ResNet50 after data augmentation==============
model = ResNet()
path_results = './Results_img/ResNet50_aug.png'
model.train(train_aug_batch,valid_aug_batch,path=path_results)
acc_ResNet_aug_score = model.test(test_aug_batch)
print(f'The marco F1 score of ResNet50_aug model is: {acc_ResNet_aug_score}')


#============================= Vision Transformer Model =============================
#============== ViT_b32 ==============
model = Vit_b32()
path_results = './Results_img/ViT_b32.png'
model.train(train_batch,valid_batch,path=path_results)
acc_Vit_b32_score = model.test(test_batch)
print(f'The marco F1 score of ViT_b32 model is: {acc_Vit_b32_score}')

#============== ViT_b32 after data augmentation ==============
model = Vit_b32()
path_results = './Results_img/ViT_b32_aug.png'
model.train(train_aug_batch,valid_aug_batch,path=path_results)
acc_Vit_b32_aug_score = model.test(test_aug_batch)
print(f'The marco F1 score of ViT_b32_aug model is: {acc_Vit_b32_aug_score}')

#============== ViT_b32 with class_weight ==============
model = Vit_b32()
path_results = './Results_img/ViT_b32_weighted.png'
model.train_weights(train_aug_batch,valid_aug_batch,path=path_results)
acc_Vit_b32_aug_weighted_score = model.test(test_aug_batch)
print(f'The marco F1 score of ViT_b32_weighted model is: {acc_Vit_b32_aug_weighted_score}')

#============== ViT_b16 after data augmentation ==============
model = Vit_b16()
path_results = './Results_img/ViT_b16_aug.png'
model.train(train_aug_batch,valid_aug_batch,path=path_results)
acc_Vit_b16_aug_score = model.test(test_aug_batch)
print(f'The marco F1 score of ViT_b16_aug model is: {acc_Vit_b16_aug_score}')


#============================= Ensemble Model =============================
model = Ensemble_model()
acc_ensemble_score = model.test(test_batch)
print(f'The marco F1 score of Ensemble model is: {acc_ensemble_score}')


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


train_labels, test_labels = loading_and_splitting_data(dataset,dataset_test)
train_batch, valid_batch, test_batch, class_weights = data_generator(train_labels_path, test_labels_path)
train_aug_batch, valid_aug_batch, test_aug_batch, class_weights = data_augmentation(train_labels_path, test_labels_path)


model = Base_CNN()
path_results = './results_img/face_shape_cnn.jpg'
model.train(train_batch,valid_batch,path=path_results)
acc_Base_CNN_score = model.test(test_batch)
print(f'The marco F1 score of CNN model on TaskB1 is: {acc_Base_CNN_score}')




model = Base_CNN_increased_layer()
path_results = './results_img/face_shape_cnn.jpg'
model.train(train_batch,valid_batch,path=path_results)
acc_Base_CNN_increased_layer_score = model.test(test_batch)
print(f'The marco F1 score of CNN model on TaskB1 is: {acc_Base_CNN_increased_layer_score}')



model = VGG_feature_extractor()
path_results = './results_img/face_shape_cnn.jpg'
# model.train(train_batch,valid_batch,path=path_results)
acc_VGG_feature_extractor_score = model.test(test_batch)
print(f'The marco F1 score of CNN model on TaskB1 is: {acc_VGG_feature_extractor_score}')


model = VGG_finetune()
path_results = './results_img/face_shape_cnn.jpg'
model.train(train_batch,valid_batch,path=path_results)
acc_VGG_fine_tune_score = model.test(test_batch)
print(f'The marco F1 score of CNN model on TaskB1 is: {acc_VGG_fine_tune_score}')


model = VGG_finetune()
path_results = './results_img/face_shape_cnn.jpg'
model.train_weights(train_batch,valid_batch, class_weights, path=path_results )
acc_VGG_fine_tune_weighted_score = model.test(test_batch)
print(f'The marco F1 score of CNN model on TaskB1 is: {acc_VGG_fine_tune_weighted_score}')


model = VGG_scratch()
path_results = './results_img/face_shape_cnn.jpg'
model.train(train_batch,valid_batch,path=path_results)
acc_VGG_fine_tune_score = model.test(test_batch)
print(f'The marco F1 score of CNN model on TaskB1 is: {acc_VGG_fine_tune_score}')


model = ResNet()
path_results = './results_img/face_shape_cnn.jpg'
model.train(train_batch,valid_batch,path=path_results)
acc_ResNet_lrfn_score = model.test(test_batch)
print(f'The marco F1 score of CNN model on TaskB1 is: {acc_ResNet_lrfn_score}')

model = ResNet()
path_results = './results_img/face_shape_cnn.jpg'
model.train(train_aug_batch,valid_aug_batch,path=path_results)
acc_ResNet_aug_score = model.test(test_aug_batch)
print(f'The marco F1 score of CNN model on TaskB1 is: {acc_ResNet_aug_score}')


model = Vit_b32()
path_results = './results_img/face_shape_cnn.jpg'
model.train(train_batch,valid_batch,path=path_results)
acc_Vit_b32_score = model.test(test_batch)
print(f'The marco F1 score of CNN model on TaskB1 is: {acc_Vit_b32_score}')

model = Vit_b32()
path_results = './results_img/face_shape_cnn.jpg'
model.train(train_aug_batch,valid_aug_batch,path=path_results)
acc_Vit_b32_aug_score = model.test(test_aug_batch)
print(f'The marco F1 score of CNN model on TaskB1 is: {acc_Vit_b32_aug_score}')

model = Vit_b32()
path_results = './results_img/face_shape_cnn.jpg'
model.train_weights(train_aug_batch,valid_aug_batch,path=path_results)
acc_Vit_b32_aug_weighted_score = model.test(test_aug_batch)
print(f'The marco F1 score of CNN model on TaskB1 is: {acc_Vit_b32_aug_weighted_score}')

model = Vit_b16()
path_results = './results_img/face_shape_cnn.jpg'
model.train(train_aug_batch,valid_aug_batch,path=path_results)
acc_Vit_b16_aug_score = model.test(test_aug_batch)
print(f'The marco F1 score of CNN model on TaskB1 is: {acc_Vit_b16_aug_score}')



model = Ensemble_model()
acc_ensemble_score = model.test(test_batch)
print(f'The marco F1 score of CNN model on TaskB1 is: {acc_ensemble_score}')


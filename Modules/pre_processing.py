"""pre_processing contains the process of loading and splitting dataset, using data generator to read data by batches,
 doing data augmentation to imitate "increase the size of data" to make model more robust.

"""

import numpy as np
import pandas as pd
import os
import random
random.seed(10)
import shutil
import tensorflow as tf
from sklearn.utils import class_weight
from keras.preprocessing.image import ImageDataGenerator




label = './Dataset/train.csv'
dataset = './Dataset/train_images/'
dataset_test = './Dataset/test_images/'
train_labels_path = './Dataset/train_labels.csv'
test_labels_path = './Dataset/test_labels.csv'




def mkdir(img_path_test):
    """
    create a new folder for storing test set
    Args:
        img_path_test: path  to store test set
    """
    folder = os.path.exists(img_path_test)
    if not folder:  # Determine if a folder exists if not then create as a folder
        os.makedirs(img_path_test)  # makedirs creates the path if it does not exist when creating the file
        print("Successfully make a new directory")
    else:
        print('The test_images folder exists')




def loading_and_splitting_data(img_path, img_path_test, test_ratio = 0.10):
    """
    Loading, splittig and ordering data
    Args:
        img_path: path to store train set
        img_path_test: path to store test set
    Returns:
        train_label: return a df contains the label of train set
        test_label: return a df contains the label of test set
    """
    mkdir(img_path_test)
    print('# Loading and transforming dataset ========')
    print('# Randomly dividing data into two directory ====')
    # Store the names of the images in the original dataset in a list
    files = sorted(os.listdir(img_path), key=lambda x: int(x.split(".")[0]))
    print(f"length of original files is : {len(files)}")
    # A random selection from the files is used as the test set
    files_test = sorted(random.sample(files ,round(test_ratio*len(files))),key=lambda x: int(x.split(".")[0]))
    print(f"length of test files is : {len(files_test)}")
    # #Move the randomly selected test to img_path_test
    for f in  files_test:
        shutil.move(img_path+f,img_path_test)
    print(len(os.listdir(img_path_test)))
    print(len(os.listdir(img_path)))

    print("loading label file and clean the data ====")
    # # Read the label file, get the face_shape in the sample, and convert it to a list
    labels = pd.read_csv(label)
    # print(labels)
    # The data in the column where the label is located is not a string type and needs to be converted
    labels['image_id'] = labels['image_id'].astype(str)
    labels['label'] = labels['label'].astype(str)
    # List of all the images within the training and testing folders
    # Iterate through the list of addresses in the training set, the test set, and obtain a list of images in the order of the paths, respectively
    # labels['file_name'] == i return the corresponding index, so after traversing the list of addresses you can return the index of each address in the label (df)
    test_index = [labels[labels['image_id'] == i].index[0]
                  for i in os.listdir(img_path_test)]
    training_index = [labels[labels['image_id'] == i].index[0]
                      for i in os.listdir(img_path)]

    train_labels = labels.iloc[[i for i in training_index]]
    test_labels = labels.iloc[[i for i in test_index]]

    print('files and labels order')
    print(train_labels)
    print(os.listdir(img_path))
    print(test_labels)
    print(os.listdir(img_path_test))
    train_labels.to_csv(train_labels_path)
    test_labels.to_csv(test_labels_path)

    return train_labels, test_labels


def data_generator(train_labels_path,test_labels_path):
    """
    Loading data by batches
    Args:
        train_labels_path: path to store label of training set
        test_labels_path: path to store label of test set
    Returns:
        train_batch: return training data by batch
        valid_batch: return validation data by batch
        test_batch: return test data by batch
        class_weights: return the weigths of imbalanced dataset
    """
    train_labels = pd.read_csv(train_labels_path, usecols=[1, 2])
    train_labels['image_id'] = train_labels['image_id'].astype(str)
    train_labels['label'] = train_labels['label'].astype(str)
    test_labels = pd.read_csv(test_labels_path, usecols=[1, 2])
    test_labels['image_id'] = test_labels['image_id'].astype(str)
    test_labels['label'] = test_labels['label'].astype(str)

    # ,validation_split = 1/9
    datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=1 / 9)
    train_batch = datagen.flow_from_dataframe(
        dataframe=train_labels,
        directory=dataset,
        x_col='image_id',
        y_col='label',
        subset='training',
        target_size=(224, 224),
        color_mode='rgb',
        batch_size=256)

    valid_batch = datagen.flow_from_dataframe(
        dataframe=train_labels,
        directory=dataset,
        x_col='image_id',
        y_col='label',
        subset='validation',
        target_size=(224, 224),
        color_mode='rgb',
        batch_size=256)

    ##这个地方要用shuffle = false 否则generator会打乱 导致顺序异常
    # Set this to False(For Test generator only, for others set True), because you need to yield the images in “order”, to predict the outputs and match them with their unique ids or filenames.
    testdatagen = ImageDataGenerator(rescale=1.0 / 255)
    test_batch = testdatagen.flow_from_dataframe(
        dataframe=test_labels,
        directory=dataset_test,
        x_col='image_id',
        y_col='label',
        shuffle=False,
        target_size=(224, 224),
        color_mode='rgb',
        batch_size=256)
    class_weights = class_weight.compute_class_weight('balanced',
                                                      classes=np.unique(train_labels['label']),
                                                      y=train_labels['label'])
    class_weights = dict(enumerate(class_weights))
    print(class_weights)

    return train_batch, valid_batch, test_batch, class_weights

def data_augmentation(train_labels_path,test_labels_path ):
    """
    Loading data with data augmentation by batches
    Args:
        train_labels_path: path to store label of training set
        test_labels_path: path to store label of test set
    Returns:
        train_batch: return training data after data augmentation  by batch
        valid_batch: return validation data after data augmentation by batch
        test_batch: return test data after data augmentation by batch
        class_weights: return the weights of imbalanced dataset
    """

    train_labels = pd.read_csv(train_labels_path, usecols=[1, 2])
    train_labels['image_id'] = train_labels['image_id'].astype(str)
    train_labels['label'] = train_labels['label'].astype(str)
    test_labels = pd.read_csv(test_labels_path, usecols=[1, 2])
    test_labels['image_id'] = test_labels['image_id'].astype(str)
    test_labels['label'] = test_labels['label'].astype(str)

    # ,validation_split = 1/9
    datagen = ImageDataGenerator(rescale=1.0 / 255,
                                 validation_split=1 / 9,
                                 samplewise_center=True,
                                 samplewise_std_normalization=True,
                                 preprocessing_function=data_augment)

    train_batch = datagen.flow_from_dataframe(
        dataframe=train_labels,
        directory=dataset,
        x_col='image_id',
        y_col='label',
        subset='training',
        target_size=(224, 224),
        color_mode='rgb',
        batch_size=64)

    valid_batch = datagen.flow_from_dataframe(
        dataframe=train_labels,
        directory=dataset,
        x_col='image_id',
        y_col='label',
        subset='validation',
        target_size=(224, 224),
        color_mode='rgb',
        batch_size=64)

    ##这个地方要用shuffle = false 否则generator会打乱 导致顺序异常
    # Set this to False(For Test generator only, for others set True), because you need to yield the images in “order”, to predict the outputs and match them with their unique ids or filenames.
    testdatagen = ImageDataGenerator(rescale=1.0 / 255,
                                     samplewise_center=True,
                                     samplewise_std_normalization=True,
                                     preprocessing_function=data_augment)
    test_batch = testdatagen.flow_from_dataframe(
        dataframe=test_labels,
        directory=dataset_test,
        x_col='image_id',
        y_col='label',
        shuffle=False,
        target_size=(224, 224),
        color_mode='rgb',
        batch_size=64)

    class_weights = class_weight.compute_class_weight('balanced',
                                                      classes=np.unique(train_labels['label']),
                                                      y=train_labels['label'])
    class_weights = dict(enumerate(class_weights))
    print(class_weights)

    return train_batch, valid_batch, test_batch, class_weights


def data_augment(image):
    """
    Transformations of the process of data augmentation
    Args:
        image: original sample from dataset
    Returns:
        image: sample after data augmentation
    """
    p_spatial = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_rotate = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_pixel_1 = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_pixel_2 = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_pixel_3 = tf.random.uniform([], 0, 1.0, dtype=tf.float32)

    # Flips
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    if p_spatial > .75:
        ##转置张量
        image = tf.image.transpose(image)

    # Rotates 随机旋转图像一定角度; 改变图像内容的朝向;
    if p_rotate > .75:
        image = tf.image.rot90(image, k=3)  # rotate 270º
    elif p_rotate > .5:
        image = tf.image.rot90(image, k=2)  # rotate 180º
    elif p_rotate > .25:
        image = tf.image.rot90(image, k=1)  # rotate 90º

    # Pixel-level transforms
    if p_pixel_1 >= .4:
        # 调整饱和度
        image = tf.image.random_saturation(image, lower=.7, upper=1.3)
    if p_pixel_2 >= .4:
        # 调整对比度
        image = tf.image.random_contrast(image, lower=.8, upper=1.2)
    if p_pixel_3 >= .4:
        # 调整亮度
        image = tf.image.random_brightness(image, max_delta=.1)

    return image






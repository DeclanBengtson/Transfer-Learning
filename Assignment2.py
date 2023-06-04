# setup
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import Sequential
from enum import Enum

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import glob
tf.config.run_functions_eagerly(True)
 
def my_team():
    '''
    Return the list of the team members of this assignment submission as a list
    of triplet of the form (student_number, first_name, last_name)
    
    '''
    return [(10414291, 'Byron', 'Chiu'), (11079550, 'Declan', 'Bengtson')]

def plot_images(x, y):
    '''
    Function utilised for the plotting of images
    
    '''
    figures = plt.figure(figsize=[15, 18])
    for i in range(50):
        ax = figures.add_subplot(5, 10, i + 1)
        ax.imshow((x[i,]*255).astype(np.uint8))
        ax.set_title(y[i])
        ax.axis('off')
        
def load_directory(dir_with_images, folder):
    '''
    Loads the images from each directoy and appends them to lists
    
    '''
    #Update the file path
    path = os.path.join(dir_with_images,folder)
    #Find all images in the directory
    files = glob.glob(os.path.join(path, '*.jpg'))
    #Initialise list
    x = []
    y =[]
    #Loop through the images, standarise it then add it to the list
    for f in files:
        image = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB).astype("float32") / 255.0
        x.append(tf.image.resize(image, (224,224)))
        if folder == 'daisy':
            y.append(int('0000'))
        if folder == 'dandelion':
            y.append(int('0001'))
        if folder == 'roses':
            y.append(int('0002'))
        if folder == 'sunflowers':
            y.append(int('0003'))
        if folder == 'tulips':
            y.append(int('0004'))
       
    #Return numpy array
    return np.array(x,dtype=object), np.array(y)

def add_data(training_x, training_y, testing_x, testing_y, val_x, val_y ,data_x,data_y):
    '''
    Splits the data into the training testing and validation sets
    
    '''
    #Split the data up into training, validation and testing
    training_x.extend(data_x[:160])
    testing_x.extend(data_x[180:])
    val_x.extend(data_x[160:180])
    training_y.extend(data_y[:160])
    testing_y.extend(data_y[180:])
    val_y.extend(data_y[160:180])
    return training_x, training_y, testing_x, testing_y, val_x, val_y 

def load_data():
    '''
    Function performs the transfer learning
    
    '''
    #Open directory with flowers dataset
    dir_with_images = os.path.join(os.getcwd(),'small_flower_dataset')
    
    #Save the directory path
    path = os.listdir(dir_with_images)

    #Initialise the lists
    training_x = []
    training_y = []
    testing_x = []
    testing_y = []
    val_x =[]
    val_y = []
    
    #Load the data into list for each directory
    daisy_x, daisy_y = load_directory(dir_with_images, path[0])
    dandelion_x, dandelion_y = load_directory(dir_with_images, path[1])
    roses_x, roses_y = load_directory(dir_with_images, path[2])
    sunflowers_x, sunflowers_y = load_directory(dir_with_images, path[3])
    tulips_x, tulips_y = load_directory(dir_with_images, path[4])
    
    #Split the data into training, validation and test then add it into a list
    training_x, training_y, testing_x, testing_y, val_x, val_y = add_data(training_x, training_y, testing_x, testing_y, val_x, val_y, daisy_x, daisy_y)
    training_x, training_y, testing_x, testing_y, val_x, val_y = add_data(training_x, training_y, testing_x, testing_y, val_x, val_y, dandelion_x, dandelion_y)
    training_x, training_y, testing_x, testing_y, val_x, val_y = add_data(training_x, training_y, testing_x, testing_y, val_x, val_y, roses_x, roses_y)
    training_x, training_y, testing_x, testing_y, val_x, val_y = add_data(training_x, training_y, testing_x, testing_y, val_x, val_y, sunflowers_x, sunflowers_y)
    training_x, training_y, testing_x, testing_y, val_x, val_y = add_data(training_x, training_y, testing_x, testing_y, val_x, val_y, tulips_x, tulips_y)
    
    #Return a numpy array of training, validation and testing
    return np.array(training_x,dtype=object),np.array(training_y,dtype=object), np.array(testing_x,dtype=object), np.array(testing_y,dtype=object),np.array(val_x,dtype=object) ,np.array(val_y,dtype=object) 

    #initiates the MobileNetV2 architecture
    MobileNetV2 = tf.keras.applications.MobileNetV2(
        input_shape=(None), # Det to none as we use the same shape tuple
        alpha=1.0,          # Default number of filters from the paper are used at each layer.
        include_top=True,   # Includes the fully-connected layer at the top of the network.
        weights="imagenet", # File to be loaded
        input_tensor=None,  # Keras tensor to use as image input for the model set to none. 
        pooling=None,       # None means that the output of the model will be the 4D tensor output of the last convolutional block.
        classes=1000,       # Integer number of classes to classify images into
        classifier_activation="softmax") 
    
    for layer in MobileNetV2.layers[:]:
        layer.trainable = False
    
    tf.keras.optimizers.SGD(
        learning_rate=0.5, momentum=0.0, nesterov=False, name="SGD")
    
    flower_model = Sequential()
    
    flower_model.add(MobileNetV2)
    flower_model.add(layers.Flatten())
    flower_model.add(layers.Dense(1024, activation="relu"))
    flower_model.add(layers.Dense(5,activation="softmax"))
    
    flower_model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])
    
    # Outputs a summary of the model
    flower_model.summary()
    
    #Load data
    training_x, training_y, testing_x, testing_y, val_x, val_y = load_data()
    
    #plot_images(test_x,test_y)
    training_y = np.reshape(training_y,(len(training_y),-1))
    testing_y = np.reshape(testing_y,(len(testing_y),-1))
    val_y = np.reshape(val_y,(len(val_y),-1))
    
    
    training_y = keras.utils.to_categorical(training_y, 5)
    testing_y = keras.utils.to_categorical(testing_y, 5)
    val_y = keras.utils.to_categorical(val_y, 5)
    training_x = np.asarray(training_x).astype('float32')
    training_y = np.asarray(training_y).astype('float32')
    testing_x = np.asarray(testing_x).astype('float32')
    testing_y = np.asarray(testing_y).astype('float32')
    val_x = np.asarray(val_x).astype('float32')
    val_y = np.asarray(val_y).astype('float32')
    
    
    print(training_x.shape)
    print(training_y.shape)
    print(testing_x.shape)
    print(testing_y.shape)
    print(val_x.shape)
    print(val_y.shape)
    
    # Outputs the model of the training data
    flower_model.fit(training_x, training_y, batch_size = 32, epochs = 10, validation_split =0.2)
    
    his = flower_model.evaluate(val_x,val_y)
    print('training?', his)
    
    his = flower_model.evaluate(testing_x,testing_y)
    print('training?', his)
    

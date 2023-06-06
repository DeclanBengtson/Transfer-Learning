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
    
    Returns:
    - team_members (list): List of team members as triplets.
    '''
    return [(10414291, 'Byron', 'Chiu'), (11079550, 'Declan', 'Bengtson')]

def plot_images(x, y):
    '''
    Function utilized for plotting images.
    
    Args:
    - x (numpy.ndarray): Input images.
    - y (numpy.ndarray): Target labels.
    '''
    figures = plt.figure(figsize=[15, 18])
    for i in range(50):
        ax = figures.add_subplot(5, 10, i + 1)
        ax.imshow((x[i,]*255).astype(np.uint8))
        ax.set_title(y[i])
        ax.axis('off')

def load_directory(dir_with_images, folder):
    '''
    Loads the images from each directory and appends them to lists.
    
    Args:
    - dir_with_images (str): Path to the directory containing images.
    - folder (str): Name of the folder/directory.
    
    Returns:
    - x (numpy.ndarray): Array of images.
    - y (numpy.ndarray): Array of labels.
    '''
    # Update the file path
    path = os.path.join(dir_with_images,folder)
    # Find all images in the directory
    files = glob.glob(os.path.join(path, '*.jpg'))
    # Initialize lists
    x = []
    y =[]
    # Loop through the images, standardize them, and add them to the lists
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
       
    # Return numpy arrays
    return np.array(x,dtype=object), np.array(y)

def add_data(training_x, training_y, testing_x, testing_y, val_x, val_y ,data_x,data_y):
    '''
    Splits the data into the training, testing, and validation sets.
    
    Args:
    - training_x (list): List of training images.
    - training_y (list): List of training labels.
    - testing_x (list): List of testing images.
    - testing_y (list): List of testing labels.
    - val_x (list): List of validation images.
    - val_y (list): List of validation labels.
    - data_x (list): List of images to be added to training, testing, and validation.
    - data_y (list): List of labels to be added to training, testing, and validation.
    
    Returns:
    - training_x (list): Updated list of training images.
    - training_y (list): Updated list of training labels.
    - testing_x (list): Updated list of testing images.
    - testing_y (list): Updated list of testing labels.
    - val_x (list): Updated list of validation images.
    - val_y (list): Updated list of validation labels.
    '''
    # Split the data up into training, validation, and testing
    training_x.extend(data_x[:160])
    testing_x.extend(data_x[180:])
    val_x.extend(data_x[160:180])
    training_y.extend(data_y[:160])
    testing_y.extend(data_y[180:])
    val_y.extend(data_y[160:180])
    return training_x, training_y, testing_x, testing_y, val_x, val_y 

#Task
def load_data():
    '''
    Function performs the transfer learning.
    
    Returns:
    - training_x (numpy.ndarray): Array of training images.
    - training_y (numpy.ndarray): Array of training labels.
    - testing_x (numpy.ndarray): Array of testing images.
    - testing_y (numpy.ndarray): Array of testing labels.
    - val_x (numpy.ndarray): Array of validation images.
    - val_y (numpy.ndarray): Array of validation labels.
    '''
    # Open directory with flowers dataset
    dir_with_images = os.path.join(os.getcwd(),'small_flower_dataset')
    
    # Save the directory path
    path = os.listdir(dir_with_images)

    # Initialize the lists
    training_x = []
    training_y = []
    testing_x = []
    testing_y = []
    val_x =[]
    val_y = []
    
    # Load the data into lists for each directory
    daisy_x, daisy_y = load_directory(dir_with_images, path[0])
    dandelion_x, dandelion_y = load_directory(dir_with_images, path[1])
    roses_x, roses_y = load_directory(dir_with_images, path[2])
    sunflowers_x, sunflowers_y = load_directory(dir_with_images, path[3])
    tulips_x, tulips_y = load_directory(dir_with_images, path[4])
    
    # Task 4: Prepare training, validation, and test sets
    # Split the data into training, validation, and test sets, then add it into a list
    training_x, training_y, testing_x, testing_y, val_x, val_y = add_data(training_x, training_y, testing_x, testing_y, val_x, val_y, daisy_x, daisy_y)
    training_x, training_y, testing_x, testing_y, val_x, val_y = add_data(training_x, training_y, testing_x, testing_y, val_x, val_y, dandelion_x, dandelion_y)
    training_x, training_y, testing_x, testing_y, val_x, val_y = add_data(training_x, training_y, testing_x, testing_y, val_x, val_y, roses_x, roses_y)
    training_x, training_y, testing_x, testing_y, val_x, val_y = add_data(training_x, training_y, testing_x, testing_y, val_x, val_y, sunflowers_x, sunflowers_y)
    training_x, training_y, testing_x, testing_y, val_x, val_y = add_data(training_x, training_y, testing_x, testing_y, val_x, val_y, tulips_x, tulips_y)
    
    
    # Return numpy arrays of training, validation, and testing data
    return np.array(training_x, dtype=object), np.array(training_y, dtype=object), np.array(testing_x, dtype=object), np.array(testing_y, dtype=object), np.array(val_x, dtype=object) ,np.array(val_y, dtype=object) 

def main():
    '''
    Main function to perform transfer learning using the MobileNetV2 architecture.
    '''
    # Load data
    training_x, training_y, testing_x, testing_y, val_x, val_y = load_data()

    #Task 5: Compile and train the model
    # Initiates the MobileNetV2 architecture
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
        learning_rate=0.6, momentum=0, nesterov=False, name="SGD")
    
    flower_model = Sequential()
    
    flower_model.add(MobileNetV2)
    flower_model.add(layers.Flatten())
    flower_model.add(layers.Dense(1024, activation="relu"))
    flower_model.add(layers.Dense(5, activation="softmax"))
    
    flower_model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])
    
    # Output a summary of the model
    flower_model.summary()
    
    # Convert labels to categorical
    training_y = np.reshape(training_y, (len(training_y), -1))
    testing_y = np.reshape(testing_y, (len(testing_y), -1))
    val_y = np.reshape(val_y, (len(val_y), -1))
    
    training_y = keras.utils.to_categorical(training_y, 5)
    testing_y = keras.utils.to_categorical(testing_y, 5)
    val_y = keras.utils.to_categorical(val_y, 5)
    
    # Convert data to appropriate type
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
    
    # Fit the model and obtain the training history
    history = flower_model.fit(training_x, training_y, batch_size=32, epochs=10, validation_split=0.2)
    
    # Call the plot_metrics function with the training history
    plot_metrics(history)
    
    # Evaluate the model on the validation and testing sets
    val_loss, val_accuracy = flower_model.evaluate(val_x, val_y)
    print('Validation Loss:', val_loss)
    print('Validation Accuracy:', val_accuracy)
    
    test_loss, test_accuracy = flower_model.evaluate(testing_x, testing_y)
    print('Testing Loss:', test_loss)
    print('Testing Accuracy:', test_accuracy)

#Task 6
def plot_metrics(history):
    '''
    Plots the training and validation metrics (loss and accuracy) over time
    
    Args:
    - history: Keras history object containing the training metrics
    
    '''
    # Extract the training and validation metrics from the history object
    training_loss = history.history['loss']
    validation_loss = history.history['val_loss']
    training_accuracy = history.history['accuracy']
    validation_accuracy = history.history['val_accuracy']

    # Create an array for the x-axis (epochs)
    epochs = np.arange(1, len(training_loss) + 1)

    # Plot training and validation losses
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, training_loss, label='Training Loss')
    plt.plot(epochs, validation_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Plot training and validation accuracies
    plt.subplot(1, 2, 2)
    plt.plot(epochs, training_accuracy, label='Training Accuracy')
    plt.plot(epochs, validation_accuracy, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    # Display the plots
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

# Flower Image Classification Project

This project involves flower image classification using transfer learning with the MobileNetV2 architecture. The goal is to develop a model that can accurately classify images of flowers into different categories.

## Setup

Before running the project, make sure to install the required libraries:
```
pip install scikit-learn numpy tensorflow opencv-python matplotlib
```
The project uses the following main components:

- **Scikit-learn**: For loading the flower dataset, training a classifier, and evaluating performance.
- **TensorFlow and Keras**: For implementing the MobileNetV2 architecture and performing transfer learning.
- **OpenCV**: For loading and processing images.
- **Matplotlib**: For visualizing images and training/validation metrics.

## Team Members

- **Byron Chiu**
  - Student Number: 10414291
  - First Name: Byron
  - Last Name: Chiu

- **Declan Bengtson**
  - Student Number: 11079550
  - First Name: Declan
  - Last Name: Bengtson

## Functions

### `my_team()`

Returns a list of team members in the format `(student_number, first_name, last_name)`.

### `plot_images(x, y)`

Function to plot images given input images `x` and target labels `y`.

### `load_directory(dir_with_images, folder)`

Loads images from a specified directory and appends them to lists.

### `add_data(training_x, training_y, testing_x, testing_y, val_x, val_y, data_x, data_y)`

Splits data into training, testing, and validation sets and adds it to the corresponding lists.

### `load_data()`

Loads the flower dataset, prepares training, testing, and validation sets, and returns them as numpy arrays.

### `main()`

The main function that performs transfer learning using the MobileNetV2 architecture. It compiles, trains, and evaluates the model.

### `plot_metrics(history)`

Plots training and validation metrics (loss and accuracy) over epochs.

## Instructions

1. Run the `main()` function to execute the transfer learning process.
2. Ensure the required libraries are installed using the provided `pip install` command.
3. Check the results, including training/validation loss, accuracy, and plotted metrics.

## Notes

- The MobileNetV2 model is used for transfer learning.
- The flower dataset is split into training, testing, and validation sets.
- Model training progress is visualized using training and validation metrics.

# Transfer-Learning
As part of a University course for Artificial Intelligence, in teams of two, we were required to develop a flower classifier using transfer learning on a neural network trained on the ImageNet dataset. <br/>
This model uses MobileNetV2 as the neural network architecture as it is the smallest of the available pretrained networks in keras. The project aimed to complete the following tasks:<br/>
&ensp;&ensp;  •	Task 1: Using the tf.keras.applications module download a pretrained MobileNetV2
network<br/>
&ensp;&ensp;  •	Task 2: Replace the last layer of the downloaded neural network with a Dense layer of the
appropriate shape for the 5 classes of the small flower dataset<br/>
&ensp;&ensp;  •	Task 3: Prepare the training, validation and test sets for the non-accelerated version of
transfer learning. <br/>
&ensp;&ensp;  •	Task 3: Compile and train the model with an SGD3 optimizer using the following parameters learning_rate=0.01, momentum=0.0, nesterov=False.<br/>
&ensp;&ensp;  •	Task 4: Plot the training and validation errors vs time as well as the training and validation
accuracies<br/>
&ensp;&ensp;  •	Task 5: Experiment with 3 different orders of magnitude for the learning rate.<br/>
&ensp;&ensp;  •	Task 5: With the best learning rate found in the previous task, add a non zero
momentum to the training with the SGD optimizer (consider 3 values for the momentum). <br/>

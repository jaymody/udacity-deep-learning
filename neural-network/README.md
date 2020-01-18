# Creating A Neural Network From Scratch

This is the first project in the Udacity Deep Learning course. I did this project before learning to use Keras, or Tensorflow, which would've made tackling the problem significantly easier, however the project was designed so you cant use those APIs and you had to use your fundamental knowledge of multiplayer-perceptrons to create the network. The network was created using numpy and its related operations and the data processing is done using pandas.

The activation functions, back propogation algorthims, weight updates, gradient descent, feed forward matrix multiplication and almost everything else involving MLPs was newly created.

The objective of the network is to predict how many customers will rent from a bike shop given previous data. 


## Data
The data given includes many features such as date, time of day, weather, etc...

Here is example below:

![Data](/res/data.png)

Here is an example of how one of the features (time), relates to the count of customers:

![Data](/res/countvsday.png)


## Network Training
Below is a summary of the training and validation loss of the network over the training period.

![Data](/res/loss.png)


## Results
Below is a comparison on a test set of the actual data for the count of customers, and what the network predicted.

![Data](/res/prediction.png)
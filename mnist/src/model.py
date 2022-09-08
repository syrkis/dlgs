# model.py
#   This file contains the Model class, which is used to create and train the model.
# by: Noah Syrkis


"""
important concepts:
1. On init our model create random layers with weights, kernels and biases.
2. We have a forward method that takes in x and outputs a prediction.
3. That's it.
"""

# imports
import torch
import torch.nn as nn
import torch.nn.functional as F


# model
class Model(nn.Module):
    def __init__(self):                      # on init
        super(Model, self).__init__()        # call super class init
        self.conv1 = nn.Conv2d(1, 32, 3, 1)  # create conv layer
        self.conv2 = nn.Conv2d(32, 1, 2, 1)  # create conv layer
        self.dense1 = nn.Linear(625, 500)    # create dense layer
        self.dense2 = nn.Linear(500, 10)     # create dense layer
        # self.decode = nn.Linear(10, 784)   # for autoencoder

    def forward(self, x):              # send data through neural net
        x = self.conv1(x)              # send x through first convolution (shrinks x a bit)
        x = F.relu(x)                  # activation function
        x = self.conv2(x)              # send x through second convolution (shrinks x a bit more)
        x = x.reshape(x.shape[0], -1)  # flatten x
        x = self.dense1(x)             # send x through first dense layer
        x = F.relu(x)                  # activation function
        x = self.dense2(x)             # last layer
        # x = self.decode(x)           # re construct x from compression for auto encoder generative stuff
        return x  # x.reshape(x.shape[0], 28, 28)

    def predict(self, x):              # actually make a prediction
        x = self.forward(x)            # send x through neural net
        return torch.argmax(x, dim=1)  # predict most likely thing

# model.py
#   This file contains the Model class, which is used to create and train the model.
# by: Noah Syrkis

# imports
import torch
import torch.nn as nn
import torch.nn.functional as F


# model
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 1, 2, 1)
        self.dense1 = nn.Linear(625, 500)
        self.dense2 = nn.Linear(500, 10)
        self.lstm = nn.LSTM()

        self.decode = nn.Linear(10, 784)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = x.reshape(x.shape[0], -1)
        x = self.dense1(x)
        x = F.relu(x)
        x = self.dense2(x)

        x = self.decode(x)
        return x.reshape(x.shape[0], 28, 28)

    def predict(self, x):
        x = self.forward(x)
        return torch.argmax(x, dim=1)

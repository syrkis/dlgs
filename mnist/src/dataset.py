# dataset.py
#   This file contains the Dataset class, which is used to load and store the data.
# by: Noah Syrkis


"""
important concepts:
1. We have a Dataset instance (ds) that spits out MNIST (x, y) TENSOR pairs.
2. On init we load the data.csv file and do a little preprocessing (divide by 255).
3. We have a __len__ method that returns the length of the dataset.
"""

# imports
import torch


# mnist dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, model='mlp'):
        with open('data/data.csv', 'r') as f:
            self.data = f.readlines()[1:]
        self.data = [x.strip().split(',') for x in self.data]
        self.targets = [int(x[0]) for x in self.data]
        self.targets = torch.tensor(self.targets)
        self.inputs = [list(map(int, x[1:])) for x in self.data]
        self.inputs = torch.tensor(self.inputs, dtype=torch.float32) / 255
        if model == 'cnn':
            self.inputs = self.inputs.view(-1, 1, 28, 28)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]

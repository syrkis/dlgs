# main.py
#   This is the main file for the project. It is the entry point for the program.
# by: Noah Syrkis

# imports
from src import *
import seaborn as sns
import matplotlib.pyplot as plt
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import torch


# main
def main():
    ds = Dataset('cnn')
    loader = DataLoader(ds, batch_size=128, shuffle=True)
    model = Model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    for x, y in loader:
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
    for x, y in loader:
        pred = model.predict(x)
        print(torch.sum(y == pred) / 128)
        break




if __name__ == '__main__':
    main()
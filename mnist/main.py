# main.py
#   This is the main file for the project. It is the entry point for the program.
# by: Noah Syrkis


"""
important concepts:
1. We have a Dataset instance (ds) that spits out MNIST (x, y) pairs.
2. We have a DataLoader instance (loader) that batches and shuffles ds.
3. We have a Model instance (model) that takes in x and outputs pred.
4. We have a loss function (criterion) that takes in pred and y and outputs loss.
5. We have an optimizer (optimizer) that takes in model.parameters() and updates the params based on loss grads.
6. We have a loop that iterates through batches of data and updates the model parameters.
"""


# imports
from src import Dataset, Model  # import my stuff

import torch                             # for summing tensors
from torch import nn                     # loss function
from torch import optim                  # for parameters optimizer
from torch.utils.data import DataLoader  # for batch loading data


# main
def main():
    """would be good practice to put the code below in src/train.py in a train() function"""
    ds = Dataset('cnn')                                    # load data set (cnn) for x -> 28 x 28
    loader = DataLoader(ds, batch_size=128, shuffle=True)  # batch size of 128
    model = Model()                                        # instance of model
    criterion = nn.CrossEntropyLoss()                      # loss function
    optimizer = optim.SGD(model.parameters(), lr=0.01)     # optimization helper
    for x, y in loader:                                    # loop through batches
        optimizer.zero_grad()      # clean gradients of parameters (remember, this loop is about finder better params)
        pred = model(x)            # make prediction
        loss = criterion(pred, y)  # calculate loss with respect to prediction
        loss.backward()            # calculate gradients of model.parameters() with respect to loss
        optimizer.step()           # update parameters with respect to gradients

    # probably belongs in an accuracy() function in src/utils.py
    for x, y in loader:                    # loop through batches
        pred = model.predict(x)            # make prediction on one batch
        print(torch.sum(y == pred) / 128)  # print accuracy of batch
        break                              # stop loop

    torch.save(model.state_dict(), 'model.pth')  # save model parameters


if __name__ == '__main__':
    main()

"""
training_tools.py

Defines functions for training a model and 
function for plotting train/val accuracy/loss.

"""

import matplotlib.pyplot as plt
import torch


def train(model, dataloader, loss_fn, optimizer):
    """ 
    Defines single training epoch. 
    Returns the average training set loss and accuracy for that epoch.
    """
    size = len(dataloader.dataset)
    model.train()
    total_loss, acc = 0, 0

    for batch, (inputs, labels) in enumerate(dataloader):
        if torch.cuda.is_available():
            inputs, labels = inputs.cuda(), labels.cuda()
        
        # compute loss
        output = model(inputs)
        loss = loss_fn(output, labels)
        total_loss += loss

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # compute accuracy
        _, predicted = torch.max(output, 1)
        acc += (predicted==labels).sum()

    return total_loss.item()/size, acc.item()/size


def validate(model, dataloader, loss_fn):
    """ 
    Returns the average validation set loss,
    and accuracy for a single epoch.
    """
    size = len(dataloader.dataset)
    model.eval()
    loss, acc = 0, 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()

            output = model(inputs)
            loss += loss_fn(output, labels)

            _, predicted = torch.max(output, 1)
            acc += (predicted==labels).sum()
    
    return loss.item()/size, acc.item()/size


def fit_model(model, train_loader, val_loader, optimizer, loss_fn, epochs=5):
    """ 
    Initializes model learning. Returns a dictionary with the average 
    training loss/accuracy, and validation loss/accuracy for each epoch.
    """
    hist = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(epochs):
        tl, ta = train(model, train_loader, loss_fn, optimizer)
        vl, va = validate(model, val_loader, loss_fn)
        print('Epoch {}, Train acc={:.4f}, Val acc={:.4f}, Train loss={:.4f}, '
                'Val loss={:.4f}'.format(epoch, ta, va, tl, vl))
        hist['train_loss'].append(tl)
        hist['train_acc'].append(ta)
        hist['val_loss'].append(vl)
        hist['val_acc'].append(va)
    return hist
    

def plot_loss_acc(hist):
    """ Plots accuracy and loss metrics over epochs. """
    plt.figure(figsize=(12,6))

    # Accuracy
    plt.subplot(121)
    plt.plot(hist['train_acc'], label='Training acc')
    plt.plot(hist['val_acc'], label='Validation acc')
    plt.legend()

    # Loss
    plt.subplot(122)
    plt.plot(hist['train_loss'], label='Training loss')
    plt.plot(hist['val_loss'], label='Validation loss')
    plt.legend()

    plt.show()
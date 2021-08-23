import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

import argparse

from modules.models.Digitnet import Digitnet
from modules.models.training_tools import fit_model, plot_loss_acc

# construct argument parser
ap = argparse.ArgumentParser()
ap.add_argument('-mp', '--model_path', required=True,
    help='path to output model after training')
ap.add_argument('-lr', '--l_rate', default=1e-3)
ap.add_argument('-e', '--epochs', default=10)
ap.add_argument('-bs', '--batch_size', default=64)
args = vars(ap.parse_args())

# set the device we will be using to run the model
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# initialize learnig rate, number of epochs, and
# batch size
l_rate = float(args['l_rate'])
epochs = int(args['epochs'])
bs = int(args['batch_size'])

# download the MNIST dataset
data_train = torchvision.datasets.MNIST('./data', train=True, 
    transform=ToTensor(), download=True)
data_test = torchvision.datasets.MNIST('./data', train=False, 
    transform=ToTensor(), download=True)

# wrap training and testing datasets with data loader
train_loader = DataLoader(data_train, batch_size=bs, shuffle=True)
test_loader = DataLoader(data_test, batch_size=bs, shuffle=True)

# initialize the model, loss function and optimizer
model = Digitnet().to(DEVICE)
loss_fn = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=l_rate)

# train the model and plot loss/accuracy graphs
hist = fit_model(model, train_loader, test_loader, optimizer, 
    loss_fn, epochs=epochs)
plot_loss_acc(hist)

# save the model
torch.save(model, args['model_path'])
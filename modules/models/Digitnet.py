import torch
import torch.nn as nn

class Digitnet(nn.Module):
    def __init__(self):
        super(Digitnet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2),
            nn.Conv2d(32, 64, (3, 3), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2),
            nn.Conv2d(64, 64, (3, 3), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(576, 288),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(288, 10),
            nn.LogSoftmax(dim=1)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 576)
        x = self.classifier(x)
        return x
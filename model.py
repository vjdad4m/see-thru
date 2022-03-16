"""
model.py
Mock-up ML model, to test data loading and processing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset

import numpy as np
import pandas as pd

from PIL import Image

from tqdm import trange

class SeeThruDataset(Dataset):
    def __init__(self):
        self.data = pd.read_csv('./test_timestamps.csv')
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        x_loc = row['radar']
        y_loc = row['pose']

        x = Image.open(f'./out/radar/{x_loc}')
        x = self.transform(x).unsqueeze(0)

        y = torch.flatten(torch.tensor(np.load(f'./out/pose/{y_loc}')))

        return x, y

class SeeThruNet(nn.Module):
    def __init__(self):
        super(SeeThruNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(800, 120)
        self.fc2 = nn.Linear(120, 66)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

ds = SeeThruDataset()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'[Using {device} device]')

print('[Initializing NN]')
net = SeeThruNet()

print('[Initializing Criterion and Optimizer]')
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum = 0.9)

for epoch in trange(100):
    for i, data in enumerate(ds):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = net(inputs)

        outputs = outputs.view(-1)
        labels = labels.view(-1)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(loss)

print('Finished Training')

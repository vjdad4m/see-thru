"""
hash_human.py
ML model to determine if the radar image has a human in it or not.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset

import time

import numpy as np
import pandas as pd

from PIL import Image

from tqdm import trange

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'[Using {device} device]')

class HasHumanDataset(Dataset):
    def __init__(self):
        self.data = pd.read_csv('./hashuman.csv')
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        y = row['has_human']
        y = torch.tensor(y, dtype=torch.float32).to(device)

        x_loc = row['file']
        x = Image.open('.'+x_loc)
        x = self.transform(x).unsqueeze(0).to(device)

        return x, y


class HasHuman(nn.Module):
    def __init__(self):
        super(HasHuman, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 2, 3)
        self.fc1 = nn.Linear(648, 120)
        self.fc2 = nn.Linear(120, 6)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(6, 1)
    
    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.conv2(x)
        x = torch.flatten(x, 1)
        x = self.fc2(self.fc1(x))
        x = self.relu(x)
        x = self.fc3(x)
        return torch.sigmoid(x)

ds = HasHumanDataset()

print('[Initializing NN]')
net = HasHuman()

if device == "cuda":
    net.cuda()

print('[Initializing Criterion and Optimizer]')
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr = 0.0003)

for epoch in trange(200):
    l = 0
    for i, data in enumerate(ds):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = net(inputs)

        outputs = outputs.view(-1)
        labels = labels.view(-1)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        l += loss

    print(l / len(ds))

    torch.save(net.state_dict(), f'./model/has_human/{l}_{time.time()}.pt')
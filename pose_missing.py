"""
pose_missing.py
Neural Network program to fill in missing pose keypoints.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset

import numpy as np
import pandas as pd

import random

from tqdm import trange

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'[Using {device} device]')

class PoseDataset(Dataset):
    def __init__(self):
        self.data = pd.read_csv('./timestamps.csv')['pose']
        indices = []
        for ix, fn in enumerate(self.data):
            d = np.load(f'./out/pose/{fn}')
            if 0 in d:
                indices.append(ix)
        self.data = self.data.drop(indices)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        Y = np.load(f'./out/pose/{self.data.iloc[index]}')
        X = []
        for data in Y:
            if random.random() < 0.2:
                X.append([0, 0])
            else:
                X.append(data)
                
        X = torch.flatten(torch.tensor(np.array(X, dtype=np.float32))).to(device)
        Y = torch.flatten(torch.tensor(Y)).to(device)

        return X, Y

class PoseMissing(nn.Module):
    def __init__(self):
        super(PoseMissing, self).__init__()
        self.fc1 = nn.Linear(26, 104)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(104, 208)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(208, 26)
    
    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.fc3(x)

def SeeThruLoss(out, real):
    assert len(out) == len(real)
    loss = 0
    for i in range(len(out)//2):
        x1, y1 = out[i*2], out[i*2+1]
        x2, y2 = real[i*2], real[i*2+1]
        x1 = x1 * 640
        x2 = x2 * 640
        y1 = y1 * 480
        y2 = y2 * 480
        dist = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
        loss += dist
    return loss

ds = PoseDataset()
net = PoseMissing()

if device == "cuda":
    net.cuda()

print('[Initializing Criterion and Optimizer]')
criterion = SeeThruLoss
optimizer = optim.Adam(net.parameters(), lr = 0.0002)

for epoch in trange(200):
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

    print(inputs)
    print(outputs)
    print(labels)
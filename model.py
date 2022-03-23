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
import cv2

from PIL import Image

from tqdm import trange

import time

# import wandb
# wandb.init(project="see-thru-test", entity="vjdad4m")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'[Using {device} device]')

class SeeThruDataset(Dataset):
    def __init__(self):
        self.data = pd.read_csv('./timestamps.csv')
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        x_loc = row['radar']
        y_loc = row['pose']

        x = Image.open(f'./out/radar/{x_loc}')
        x = self.transform(x).unsqueeze(0).to(device)

        y = torch.flatten(torch.tensor(np.load(f'./out/pose/{y_loc}'))).to(device)

        return x, y, y_loc

class SeeThruNet(nn.Module):
    def __init__(self):
        super(SeeThruNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(960, 120)
        self.fc2 = nn.Linear(120, 26)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

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
    return loss # / (len(out)//2)

ds = SeeThruDataset()

print('[Initializing NN]')
net = SeeThruNet()

if device == "cuda":
    net.cuda()

print('[Initializing Criterion and Optimizer]')
criterion = SeeThruLoss # nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr = 0.0001)

# wandb.config = {"learning_rate": 0.001,"epochs": 100,"batch_size": 1}

l = 0

for epoch in trange(100):
    for i, data in enumerate(ds):
        inputs, labels, loc = data

        optimizer.zero_grad()

        outputs = net(inputs)

        outputs = outputs.view(-1)
        labels = labels.view(-1)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
            
        # wandb.log({"loss": loss})
        # wandb.watch(net)
    
    print(loss)

    img = np.array(Image.open(f"./out/img/{loc[:-4]}.png"))
    outputs = outputs.cpu()
    labels = labels.cpu()
    for i in range(len(outputs)//2):
        x1, y1 = outputs[i*2].detach(), outputs[i*2+1].detach()
        x2, y2 = labels[i*2].detach(), labels[i*2+1].detach()
        x1 = int(x1.item() * 640)
        x2 = int(x2.item() * 640)
        y1 = int(y1.item() * 480)
        y2 = int(y2.item() * 480)
        # print(x1, x2, y1, y2)
        img = cv2.circle(img, (x1, y1), 2, (255, 0, 0), 2)
        img = cv2.circle(img, (x2, y2), 2, (0, 255, 0), 2)
        img = cv2.line(img, (x1, y1), (x2, y2), (128, 128, 128), 2)

    img = Image.fromarray(img)
    img.save(f'{loss}.png')
    
    # torch.save(net.state_dict(), f'./{loss}_{time.time()}.pt')
               
print('Finished Training')

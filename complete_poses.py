"""
complete_poses.py
Predict missing (0,0)-s from every pose in ./out/poses
"""

import os

import torch
import torch.nn as nn

import numpy as np

from tqdm import tqdm

DIR = os.listdir('./out/pose')[1:]

class PoseMissing(nn.Module):
    def __init__(self):
        super(PoseMissing, self).__init__()
        self.fc1 = nn.Linear(26, 104)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(104, 26)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(52, 26)
    
    def forward(self, x):
        x_orig = x
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = torch.cat([x, x_orig], -1)
        return self.fc3(x)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'[Using {device} device]')

print('[Initializing NN]')
net = PoseMissing()

if device == "cuda":
    net.cuda()

MODEL_LOC = "./model/pose_missing/171.1840057373047_1648292100.2688458.pt"
net.load_state_dict(torch.load(MODEL_LOC))

for file in tqdm(DIR):
    pose_orig = np.load('./out/pose/' + file)
    pose_input = torch.flatten(torch.tensor(pose_orig)).to(device)
    
    with torch.no_grad():
        pose_predicted = net(pose_input).detach().cpu().numpy()
        pose_predicted = pose_predicted.reshape(-1, 2)

    pose_final = []
    for i, kp in enumerate(pose_orig):
        if kp[0] == 0 and kp[1] == 0:
            pose_final.append(pose_predicted[i])
        else:
            pose_final.append(pose_orig[i])
    
    pose_final = np.array(pose_final, dtype = np.float32)
    np.save(f'./out/pose_complete/{file}', pose_final)

print('Complete!')
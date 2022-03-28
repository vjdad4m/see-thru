"""
predict_pose.py
Calculate missing pose keypoints.
"""

import torch
import torch.nn as nn

import numpy as np
import cv2
import csv

import random

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

with open('timestamps.csv') as f:
    csv_reader = csv.reader(f, delimiter=',')
    data = list(csv_reader)[1:]
    l_data = len(data)

image = np.zeros((480, 640, 3), np.uint8)

def generate_fake(pose):
    pose_new = []
    for kp in pose:
        if random.random() < 0.2:
            pose_new.append([0, 0])
        else:
            pose_new.append(kp)
    return np.array(pose_new, dtype = np.float32)

data_idx = 45

pose_original = np.load(f'./out/pose/{data[data_idx][1]}')
pose_fake = generate_fake(pose_original)
pose_input = torch.flatten(torch.tensor(pose_fake)).to(device)

with torch.no_grad():
    pose_predicted = np.array(net(pose_input).cpu().reshape(-1, 2))

def draw_keypoints(img, kps, color):
    for kp in kps:
        x = int(kp[0] * 640)
        y = int(kp[1] * 480)
        img = cv2.circle(img, (x, y), 4, color, -1)
    return img

# img = draw_keypoints(image, pose_original, (255, 0, 0))
img = draw_keypoints(image, pose_fake, (0, 255, 0))
img = draw_keypoints(image, pose_predicted, (0, 0, 255))

cv2.imshow('output', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
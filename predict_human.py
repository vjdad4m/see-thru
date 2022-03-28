"""
predict_human.py
Predict if there is a human present on the radar image.
Based on model trained in has_human.py.
"""

from sys import displayhook
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

import numpy as np

import cv2

import WalabotAPI as wlbt

from config import Config

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (0, 50)
fontScale              = 2
fontColor              = (255,255,255)
thickness              = 1
lineType               = 2

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

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'[Using {device} device]')

print('[Initializing NN]')
net = HasHuman()

MODEL_LOC = f"./model/has_human/24.0039005279541_1648499698.723307.pt"
net.load_state_dict(torch.load(MODEL_LOC))

if device == "cuda":
    net.cuda()

wlbt.Init()
wlbt.SetSettingsFolder()
wlbt.ConnectAny()

wlbt.SetProfile(wlbt.PROF_SENSOR)
wlbt.SetArenaR(Config.MIN_R,    Config.MAX_R, Config.RES_R)
wlbt.SetArenaTheta(Config.MIN_T,Config.MAX_T, Config.RES_T)
wlbt.SetArenaPhi(Config.MIN_P,  Config.MAX_P, Config.RES_P)

wlbt.Start()

wlbt.Trigger()
print(np.array(wlbt.GetRawImageSlice()[0]).shape)

transform = transforms.ToTensor()

red = []
for x in range(255):
    row = []
    for y in range(255):
        row.append(np.array([0, 0, 255], dtype=np.uint8))
    red.append(row)
red = np.array(red, dtype=np.uint8)

blue = []
for x in range(255):
    row = []
    for y in range(255):
        row.append(np.array([255, 0, 0], dtype=np.uint8))
    blue.append(row)
blue = np.array(blue, dtype=np.uint8)

running = True
while running:
    wlbt.Trigger()
    img = wlbt.GetRawImageSlice()
    img = np.array(img[0], np.uint8)

    with torch.no_grad():
        image = transform(img).unsqueeze(0).to(device)
        outputs = net(image)

        outputs = outputs.cpu()

    if outputs > Config.HUMAN_THRESHOLD:
        display = blue
    elif outputs < Config.HUMAN_THRESHOLD:
        display = red

    display_copy = display.copy()
    
    cv2.putText(display_copy,
    str(round(float(outputs), 4)), 
    bottomLeftCornerOfText, 
    font, 
    fontScale,
    fontColor,
    thickness,
    lineType)

    cv2.imshow('display', display_copy)
    if cv2.waitKey(1) == ord('q'):
        running = False

cv2.destroyAllWindows()
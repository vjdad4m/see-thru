"""
predict.py
Load a trained NN, and run it live on a radar.
"""

from doctest import OutputChecker
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

import numpy as np
import cv2
import time

from PIL import Image

import WalabotAPI as wlbt

from config import Config

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

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'[Using {device} device]')

print('[Initializing NN]')
net = SeeThruNet()

MODEL_LOC = f"./model/{Config.MODEL_FN}"
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

cap = cv2.VideoCapture(0)

transform = transforms.ToTensor()

running = True
while running:
    t1 = time.time()
    
    wlbt.Trigger()
    img = wlbt.GetRawImageSlice()
    t_taken = time.time()

    img_raw = np.array(img[0], np.uint8)

    img = Image.fromarray(img_raw)
    
    with torch.no_grad():
        image = transform(img).unsqueeze(0).to(device)
        outputs = net(image)

        ret, img = cap.read()

        outputs = outputs.cpu()[0]

        for i in range(len(outputs)//2):
            x, y = outputs[i*2].detach(), outputs[i*2+1].detach()
            x = int(x.item() * 640)
            y = int(y.item() * 480)
            img = cv2.circle(img, (x, y), 6, (0, 0, 255), -1)

        cv2.imshow('cap', img)
        img_raw = cv2.resize(img_raw, (640, 480))
        cv2.imshow('radar', img_raw)
        
        if cv2.waitKey(1) == ord('q'):
            running = False

cv2.destroyAllWindows()
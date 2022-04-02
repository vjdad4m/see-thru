"""
create_video.py
Create a video based on timestamps.csv.
"""

import pandas
import cv2
import torch

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

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

MODEL_LOC = f"./model/see_thru/{Config.MODEL_FN}"
net.load_state_dict(torch.load(MODEL_LOC))

if device == "cuda":
    net.cuda()

transform = transforms.ToTensor()

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

def draw_keypoints(img, keypoints):
    draw = ImageDraw.Draw(img)
    for kp in keypoints:
        x, y = int(kp[0] * 640), int(kp[1] * 480)
        draw.ellipse((x-5, y-5, x+5, y+5), fill = "green")
    return img

def draw_keypoints_1d(img, keypoints):
    draw = ImageDraw.Draw(img)
    for i in range(len(keypoints) // 2):
        kp = [keypoints[i*2], keypoints[i*2+1]]
        x, y = int(kp[0] * 640), int(kp[1] * 480)
        draw.ellipse((x-5, y-5, x+5, y+5), fill = "green")
    return img

timestamps = pandas.read_csv("timestamps.csv")

fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
out = cv2.VideoWriter('VIDEO.avi', fourcc, 10, (640 * 2, 480 * 2))

def create_frame(l_img, l_pose, l_radar):
    font = ImageFont.truetype("./res/monospace.ttf", 32) # ImageFont.load_default()

    img = Image.open('./out/img/' + l_img)
    radar = Image.open('./out/radar/' + l_radar)
    img_radar = radar.resize((640, 480))

    pose = np.load('./out/pose_complete/' + l_pose)
    img_pose = Image.new("RGB", (640, 480))
    img_pose = draw_keypoints(img_pose, pose)

    with torch.no_grad():
        pose_pred = net(transform(radar).unsqueeze(0).to(device))
        pose_pred = pose_pred.cpu().numpy()[0]
    img_pred = Image.new("RGB", (640, 480))
    img_pred = draw_keypoints_1d(img_pred, pose_pred)

    frameh1 = get_concat_h(img, img_radar)
    frameh2 = get_concat_h(img_pose, img_pred)
    frame = get_concat_v(frameh1, frameh2)

    draw_frame = ImageDraw.Draw(frame)
    draw_frame.fontmode = "L"
    draw_frame.text((0, 0), "camera", font = font, fill = (0, 0, 255))
    draw_frame.text((640, 0), "radar", font = font, fill = (0, 0, 255))
    draw_frame.text((0, 480), "pose", font = font, fill = (0, 0, 255))
    draw_frame.text((640, 480), "prediction", font = font, fill = (0, 0, 255))

    frame = np.array(frame)
    return frame

for data in tqdm(timestamps.iterrows()):
    data = data[1]
    l_img = data['img']
    l_pose = data['pose']
    l_radar = data['radar']
    distance = data['distance']
    frame = create_frame(l_img, l_pose, l_radar)
    out.write(frame)

out.release()
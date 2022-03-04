#!/usr/bin/env python3

import PySimpleGUI as sg
import numpy as np
import cv2
from PIL import Image
import csv
import io

sg.theme('DarkAmber')

with open('test_timestamps.csv') as f:
    csv_reader = csv.reader(f, delimiter=',')
    data = list(csv_reader)
    l_data = len(data)

def get_img(path):
    image = Image.open(path)
    image = image.resize((256, 256))
    bio = io.BytesIO()
    image.save(bio, format='PNG')
    return bio.getvalue()

def get_pose(img, kps):
    image = Image.open(img)
    image = image.resize((256, 256))
    image = np.array(image)

    for x, y in kps:
        x = int(x * 256)
        y = int(y * 256)
        cv2.circle(image, (x, y), 4, (255, 255, 255))

    image = Image.fromarray(image)
    bio = io.BytesIO()
    image.save(bio, format='PNG')
    return bio.getvalue()

layout= [   [sg.Stretch(), sg.Button('X')],
            [sg.ProgressBar(max_value = l_data, orientation = 'h', size=(46, 20), key = 'progress')],
            [sg.HorizontalSeparator()],
            [sg.Text('Image', size=(36), justification='center'), sg.Text('Pose', size=(36), justification='center'), sg.Text('Radar', size=(36), justification='center')],
            [sg.Image(key='img', size=(256, 256)), sg.Image(key='pose', size=(256, 256)), sg.Image(key='radar', size=(256, 256))],
            [sg.Button('<<'), sg.Button('<'), sg.Button('>'), sg.Button('>>')]
        ]

window = sg.Window('See-Thru', layout, element_justification='c')

data_index = 0

while True:
    event, values = window.read()
    shouldUpdate = False
    
    if event == sg.WIN_CLOSED or event == 'X':
        break
    
    if event == '>':
        data_index = min(data_index + 1, l_data - 1)
        shouldUpdate = True
    elif event == '<':
        data_index = max(1, data_index - 1)
        shouldUpdate = True
    elif event == '>>':
        data_index = min(data_index + 10, l_data - 1)
        shouldUpdate = True
    elif event == '<<':
        data_index = max(1, data_index - 10)
        shouldUpdate = True

    if shouldUpdate:
        window['img'].update(data = get_img('./out/img/' + data[data_index][0]))
        window['pose'].update(data = get_pose('./out/img/' + data[data_index][0], np.load('./out/pose/' + data[data_index][1])))
        window['radar'].update(data = get_img('./out/radar/' + data[data_index][2]))
        window['progress'].update_bar(data_index)

window.close()

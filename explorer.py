import PySimpleGUI as sg

import io
import cv2
import numpy as np
from PIL import Image

from pose import PoseExtractor
from generate_dataset import get_matches

from supplement_poses import PoseSupplementer

sg.theme('DarkAmber')

pe = PoseExtractor()
ps = PoseSupplementer()
matches = get_matches()

l_data = len(matches)
data_index = 0

def get_data(match):
    radar = np.load('data/radar/' + match[0])['arr_0']
    cam = np.load('data/cam/' + match[1])['arr_0']
    radar_image = cv2.resize(radar, (256, 256))
    radar_image = Image.fromarray(radar_image)
    cam_image = cv2.resize(cam, (256, 256))
    cam_image = Image.fromarray(cam_image)

    _cam_image = io.BytesIO()
    cam_image.save(_cam_image, format='PNG')

    _radar_image = io.BytesIO()
    radar_image.save(_radar_image, format='PNG')

    pose_image = np.zeros((256, 256, 3), dtype=np.uint8)
    pose = pe(cam)
    if pose is not None:
        pose_complete = ps.predict(pose)
        for x, y in pose:
            x = int(x * 256)
            y = int(y * 256)
            cv2.circle(pose_image, (x, y), 4, (0, 255, 0))
        for x, y in pose_complete:
            x = int(x * 256)
            y = int(y * 256)
            cv2.circle(pose_image, (x, y), 4, (255, 0, 0))

    pose_image = Image.fromarray(pose_image)
    _pose_image = io.BytesIO()
    pose_image.save(_pose_image, format='PNG')

    return _radar_image.getvalue(), _cam_image.getvalue(), _pose_image.getvalue()
        

layout= [   [sg.Stretch(), sg.Button('X')],
            [sg.Text(key='line', text='0', size=(36), justification='center')],
            [sg.ProgressBar(max_value = l_data, orientation = 'h', size=(46, 20), key = 'progress')],
            [sg.HorizontalSeparator()],
            [sg.Text('Image', size=(36), justification='center'), sg.Text('Pose', size=(36), justification='center'), sg.Text('Radar', size=(36), justification='center')],
            [sg.Image(key='img', size=(256, 256)), sg.Image(key='pose', size=(256, 256)), sg.Image(key='radar', size=(256, 256))],
            [sg.Button('<<'), sg.Button('<'), sg.Button('>'), sg.Button('>>')]
        ]

window = sg.Window('see thru viewer', layout, element_justification='c', return_keyboard_events=True)

while True:
    event, values = window.read()
    shouldUpdate = False

    if event == sg.WIN_CLOSED or event in ('X', 'q'):
        break

    if event in ('>', 'Right:114', 'Right:39'):
        data_index = min(data_index + 1, l_data - 1)
        shouldUpdate = True
    elif event in ('<', 'Left:113', 'Left:38'):
        data_index = max(1, data_index - 1)
        shouldUpdate = True
    elif event == '>>':
        data_index = min(data_index + 10, l_data - 1)
        shouldUpdate = True
    elif event == '<<':
        data_index = max(1, data_index - 10)
        shouldUpdate = True

    if shouldUpdate:
        radar_image, cam_image, pose_image = get_data(matches[data_index])
        window['img'].update(data = cam_image)
        window['pose'].update(data = pose_image)
        window['radar'].update(data = radar_image)
        window['progress'].update_bar(data_index)
        window['line'].update(data_index)

window.close()
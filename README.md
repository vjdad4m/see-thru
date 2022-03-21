# See-Thru

This repo contains applications used by our project, which aims to create multiple machine learning applications based on radar sensor data. Specifically, it uses the [walabot](https://walabot.com/makers) radar.

* Data collection works on Windows.
* Data processing applications work on Windows, Linux and MacOS.

GUI framework used: [PySimpleGUI](https://github.com/PySimpleGUI/PySimpleGUI)

## Setup

1. Install the [WalabotSDK](https://walabot.com/getting-started) (has problems running on Linux, so using the Windows version is recommended).

2. Install dependencies using pip (`requirements.txt`)

3. For data collection run `measure.py`

4. For data exploration run `app.py`  

<img src="./res/app.png">

## Data Collection

### Hardware used

- Ausdom AF640 (Webcam)
- Vayyar Walabot Creator (Radar)

### Experiment setup

<center>
    <img src="./res/setup_1.png" width=40%>
    <img src="./res/setup_2.png" width=40%>
</center>

> Use the `measure.py` script to connect to the webcam and the radar and start data collection. 

### Storage

Dataset size is 138 Mb / 1000 samples. Meaning a dataset consisting of 1 million samples will take 135 GBs of storage space.

## Data Processing

<img src="./res/pose.png" width=256px>

Used pose detection method is [Media Pipe Pose](https://google.github.io/mediapipe/solutions/pose.html). Unnecessary keypoints are discarded (33 -> 13).
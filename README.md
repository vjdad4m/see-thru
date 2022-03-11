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
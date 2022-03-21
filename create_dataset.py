#!/usr/bin/env python3

"""
create_dataset.py
Export only useful images, poses and radar measurements.
"""

import csv
import shutil

ts = open('timestamps.csv', 'r')
csvreader = csv.reader(ts)

header = next(csvreader)

print('Starting copying from /out to /dataset')

for loc_img, loc_pose, loc_radar, dist in csvreader:
    shutil.copy('./out/img/' + loc_img, './dataset/img/' + loc_img)
    shutil.copy('./out/pose/' + loc_pose, './dataset/pose/' + loc_pose)
    shutil.copy('./out/radar/' + loc_radar, './dataset/radar/' + loc_radar)

print('Done!')
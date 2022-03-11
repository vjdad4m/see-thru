#!/usr/bin/env python3
import os
from config import Config

f_pose = os.listdir('./out/pose')
f_radar = os.listdir('./out/radar')

f_pose.remove('.gitignore')
f_radar.remove('.gitignore')

f_pose.sort()
f_radar.sort()

i_pose = 0
i_radar = 0

with open('test_timestamps.csv', 'w') as f:
    f.write('img,pose,radar,distance\n')
    while i_pose < len(f_pose) - 1 and i_radar < len(f_radar) - 1:
        c_pose = float(os.path.splitext(f_pose[i_pose])[0])
        c_radar = float(os.path.splitext(f_radar[i_radar])[0])
        
        distance = c_pose - c_radar

        if abs(distance) < Config.MATCH_DISTANCE:
            f.write(f'{f_pose[i_pose][:-4] + ".png"},{f_pose[i_pose]},{f_radar[i_radar]},{distance}\n')

        if distance > 0:
            i_radar += 1
        else:
            i_pose += 1
            

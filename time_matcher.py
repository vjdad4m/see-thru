#!/usr/bin/env python3
import os

# f_img = os.listdir('./out/img') # same as f_pose
f_pose = os.listdir('./out/pose')
f_radar = os.listdir('./out/radar')

# f_img.remove('.gitignore') # same as f_pose
f_pose.remove('.gitignore')
f_radar.remove('.gitignore')

# f_img.sort() # same as f_pose
f_pose.sort()
f_radar.sort()

i_pose = 0
i_radar = 0

# TODO: Save as csv (now using: python3 time_matcher.py > timestamps.csv)

print('img,pose,radar,distance')
while i_pose < len(f_pose) - 1 and i_radar < len(f_radar) - 1:
    c_pose = float(os.path.splitext(f_pose[i_pose])[0])
    c_radar = float(os.path.splitext(f_radar[i_radar])[0])
    
    distance = c_pose - c_radar

    # valid timestamp if distance < 0.067 (~15 fps)

    if abs(distance) < 0.067:
        print(f'{f_pose[i_pose][:-4] + ".png"},{f_pose[i_pose]},{f_radar[i_radar]},{distance}')

    if distance > 0:
        i_radar += 1
    else:
        i_pose += 1

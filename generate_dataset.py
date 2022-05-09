import os
import tqdm
import numpy as np

from pose import PoseExtractor
from config import Config
from supplement_poses import PoseSupplementer

def get_matches():
    f_cam = os.listdir('data/cam')
    f_radar = os.listdir('data/radar')

    f_cam.remove('.gitignore')
    f_radar.remove('.gitignore')

    f_cam.sort()
    f_radar.sort()

    i_cam = 0
    i_radar = 0

    matches = []

    while i_cam < len(f_cam) - 1 and i_radar < len(f_radar) - 1:
        # get time taken from filename
        time_cam = float(os.path.splitext(f_cam[i_cam])[0])
        time_radar = float(os.path.splitext(f_radar[i_radar])[0])

        distance = time_cam - time_radar
        
        # check if match
        if abs(distance) < Config.match_distance:
            matches.append((f_radar[i_radar], f_cam[i_cam]))
            # print('match', distance, f_radar[i_radar], f_cam[i_cam])
        
        if distance > 0:
            i_radar += 1
        else:
            i_cam += 1

    return matches

def generate_dataset(matches):
    pe = PoseExtractor()
    ps = PoseSupplementer()
    X, Y = [], []

    for match in tqdm.tqdm(matches):
        # read arrays
        radar = np.load(f'data/radar/{match[0]}')['arr_0']
        cam = np.load(f'data/cam/{match[1]}')['arr_0']
        pose = pe(cam) # get pose
        
        # check if image contains pose
        if pose is not None:
            if Config.fill_in_missing_keypoints:
                # fill in blank keypoints with generated ones
                pose_complete = ps.predict(pose)
                for idx, kp in enumerate(pose):
                    if kp[0] == 0 and kp[1] == 0:
                        pose[idx] = pose_complete[idx]

            X.append(radar)
            Y.append(pose)
    
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

if __name__ == '__main__':
    matches = get_matches()
    print(f'found {len(matches)} matches')
    X, Y = generate_dataset(matches)
    np.savez('processed/dataset.npz', X, Y)
    print(f'created a dataset with len {len(X)}')
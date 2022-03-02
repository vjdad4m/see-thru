#!/usr/bin/env python3
import multiprocessing

import cv2
import numpy as np
import time
from PIL import Image

from config import Config

class ProcessPose(multiprocessing.Process):
    def __init__(self, id):
        super(ProcessPose, self).__init__()
        self.id = id

    def run(self):
        import mediapipe as mp

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)

        pose = mp.solutions.pose.Pose(min_detection_confidence = 0.5, min_tracking_confidence = 0.5)
        
        fps = lambda t1, t2: 1 / (t2 - t1)
        measure_previous = time.time()
        
        while cap.isOpened():
            t1 = time.time()

            ret, frame = cap.read()
            t_taken = time.time()

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            res = pose.process(img)

            if res.pose_landmarks:
                landmarks = []
                for lm in res.pose_landmarks.landmark:
                    if lm.visibility > 0.5:
                        landmarks.append([lm.x, lm.y])
                    else:
                        landmarks.append([0, 0])

                landmarks = np.array(landmarks, np.float32)
                np.save(f'./out/pose/{t_taken}.npy', landmarks)

                if Config.DRAW_POSE:
                    pose_img = frame.copy()
                    
                    mp.solutions.drawing_utils.draw_landmarks(
                        pose_img,
                        res.pose_landmarks,
                        mp.solutions.pose.POSE_CONNECTIONS,
                        landmark_drawing_spec = mp.solutions.drawing_styles.get_default_pose_landmarks_style()
                    )

                    cv2.imshow('pose', pose_img)

            if Config.SAVE_RAW_IMG:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                # np.save(f'./out/img/{t_taken}.npy', img)
                img = Image.fromarray(img)
                img.save(f'./out/img/{t_taken}.png')

            if Config.DRAW_CAM:
                cv2.imshow('frame', frame)
            
            if cv2.waitKey(1) == ord('q'):
                break
            
            t2 = time.time()
            
            if t2 - measure_previous > Config.LOGGING_RATE:
                measure_previous = t2
                print(f'[Process Pose] fps: {fps(t1, t2)}')
  
        cv2.destroyAllWindows()



class ProcessRadar(multiprocessing.Process):
    def __init__(self, id):
        super(ProcessRadar, self).__init__()
        self.id = id

    def run(self):
        import WalabotAPI as wlbt
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
        

        fps = lambda t1, t2: 1 / (t2 - t1)
        measure_previous = time.time()
        
        while True:
            t1 = time.time()
            
            wlbt.Trigger()
            img = wlbt.GetRawImageSlice()
            t_taken = time.time()

            img = np.array(img[0], np.uint8)

            image = Image.fromarray(img)
            image.save(f'./out/radar/{t_taken}.png')
            
            if Config.DRAW_RADAR:
                cv2.imshow('radar', img)

            if cv2.waitKey(1) == ord('q'):
                break

            t2 = time.time()

            if t2 - measure_previous > Config.LOGGING_RATE:
                measure_previous = t2
                print(f'[Process Radar] fps: {fps(t1, t2)}')

        cv2.destroyAllWindows()



def main():
    p_pose = ProcessPose(0)
    p_radar = ProcessRadar(1)

    print('Starting Pose Process')
    p_pose.start()

    print('Starting Radar Process')
    p_radar.start()

if __name__ == '__main__':
    main()

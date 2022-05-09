import numpy as np
import cv2
import WalabotAPI as wlbt

class Camera:
    def __init__(self, device_id = 0, width = 640, height = 480):
        import cv2
        self.cap = cv2.VideoCapture(device_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    def get_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = np.array(frame, np.uint8)
        return frame

    def release(self):
        self.cap.release()

class Radar:
    def __init__(self, config):
        self.config = config
        wlbt.Init()
        wlbt.SetSettingsFolder()
        wlbt.ConnectAny()
        wlbt.SetProfile(wlbt.PROF_SENSOR)
        wlbt.SetArenaR(config['MinR'], config['MaxR'], config['ResR'])
        wlbt.SetArenaTheta(config['MinT'], config['MaxT'], config['ResT'])
        wlbt.SetArenaPhi(config['MinP'], config['MaxP'], config['ResP'])
        wlbt.Start()
    
    def get_frame(self):
        wlbt.Trigger()
        frame = wlbt.GetRawImageSlice()[0]
        frame = np.array(frame, np.uint8)
        return frame

    def release(self):
        wlbt.Stop()

if __name__ == '__main__':
    radar_config = {'MinR': 100, 'MaxR': 250, 'ResR': 1,
                    'MinT': -30, 'MaxT': 30, 'ResT': 5,
                    'MinP': -60, 'MaxP': 60, 'ResP': 2}
                    
    radar = Radar(radar_config)
    cam = Camera()

    while True:
        cv2.imshow('cam', cam.get_frame())
        cv2.imshow('radar', radar.get_frame())
        if cv2.waitKey(1) == ord('q'):
            cam.release()
            radar.release()
            cv2.destroyAllWindows()
            break
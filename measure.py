import cv2
import time
import numpy as np
from threading import Thread

from sensor import Camera, Radar
from config import Config

class Measurement:
    def __init__(self, m_type, data):
        self.m_type = m_type
        self.data = None
        if data is not None:
            self.data = np.array(data, np.uint8)
        self.time = round(float(time.time()), 4)

    def __repr__(self):
        return '%s measurement at %.4f :\n%s' % (self.m_type, self.time, str(self.data))

    def save(self, path=None):
        if self.data is not None:
            if not path:
                np.savez(f'data/{self.m_type}/{self.time}.npz', self.data)
            else:
                np.savez(path, self.data)

class Measurer:
    def __init__(self):
        self.camera = Camera()
        self.radar = Radar(Config.radar_config)
        self.camera_latest = Measurement(None, None)
        self.radar_latest = Measurement(None, None)

    def measure_camera(self):
        while True:
            self.camera_latest = Measurement('cam', self.camera.get_frame())
    
    def measure_radar(self):
        while True:
            self.radar_latest = Measurement('radar', self.radar.get_frame())

    def run_camera(self):
        thread_camera = Thread(target=self.measure_camera)
        thread_camera.daemon = True
        print('starting camera thread')
        thread_camera.start()

    def run_radar(self):
        thread_radar = Thread(target=self.measure_radar)
        thread_radar.daemon = True
        print('starting radar thread')
        thread_radar.start()

    def run(self):
        self.run_camera()
        self.run_radar()
        

def measure():
    m = Measurer()
    m.run()

    n_samples = 0

    i = 0
    t_start = time.time()

    while True:
        # get latest measurements
        camera_latest = m.camera_latest
        radar_latest = m.radar_latest

        if camera_latest.data is not None and radar_latest.data is not None:
            # save measurements
            if abs(camera_latest.time - radar_latest.time) < Config.match_distance:
                camera_latest.save()
                radar_latest.save()
                n_samples += 1

            # display sensors
            if Config.draw_cam:
                cv2.imshow('camera', camera_latest.data)
            if Config.draw_radar:
                cv2.imshow('radar', radar_latest.data)
            if cv2.waitKey(1) == ord('q'):
                cv2.destroyAllWindows()
                break

        # measure fps
        if i > 100:
            t_now = time.time()
            print(f'running at {1 / (t_now - t_start) * 100} fps, collected {n_samples} samples')
            t_start = t_now
            i = 0
        i += 1

        time.sleep(0.05) # sleep 50 ms
    
    m.camera.release()
    m.radar.release()
    
if __name__ == '__main__':
    measure()
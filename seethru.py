import torch
import cv2
import numpy as np

from train import SeeThruNet
from measure import Measurer

class SeeThruPredictor:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SeeThruNet()
        vals = torch.load('nets/seethru.pth')
        self.model.load_state_dict(vals)
        self.model.to(self.device)
    
    def predict(self, inputs):
        inputs = torch.tensor(np.array([inputs])).to(self.device).unsqueeze(0).float()
        with torch.no_grad():
            out = self.model(inputs)
        return out.cpu().numpy().reshape((13, 2)).astype(np.float32)

def create_keypoint_image(kps, w = 640, h = 480, c = (0, 255, 0)):
    arr = np.zeros((h, w, 3))
    for kp in kps:
        cv2.circle(arr, (int(kp[0] * w), int(kp[1] * h)), 3, c)
    return arr

def main():
    stp = SeeThruPredictor()
    m = Measurer()
    m.run_radar()
    while True:
        latest = m.radar_latest
        if latest.m_type is not None:
            prediction = stp.predict(latest.data)
            pred_image = create_keypoint_image(prediction)

            cv2.imshow('radar', latest.data)
            cv2.imshow('prediction', pred_image)

            if cv2.waitKey(1) == ord('q'):
                break

if __name__ == '__main__':
    main()
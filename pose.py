import numpy as np
import mediapipe as mp

class PoseExtractor:
    def __init__(self, min_detection_confidence = 0.5, min_tracking_confidence = 0.5):
        self.pose = mp.solutions.pose.Pose(min_detection_confidence = min_detection_confidence, min_tracking_confidence = min_tracking_confidence)

    def __call__(self, img):
        res = self.pose.process(img)
        landmarks = None

        if res.pose_landmarks:
            landmarks = []
            for lm in res.pose_landmarks.landmark:
                if lm.visibility > 0.5:
                    landmarks.append([lm.x, lm.y])
                else:
                    landmarks.append([0, 0])

            # extract specific keypoints
            landmarks = [landmarks[0], landmarks[12], landmarks[11], landmarks[14], landmarks[13], landmarks[16], landmarks[15],
                        landmarks[24], landmarks[23], landmarks[26], landmarks[25], landmarks[28], landmarks[27]]

            landmarks = np.array(landmarks, np.float32)

        return landmarks
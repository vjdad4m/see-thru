"""
config.py
Used to configure experiment parameters.
"""

class Config(object):
    # GENERAL SETTINGS
    LOGGING_RATE    = 1         # seconds
    DRAW_CAM        = False     # yes/no
    DRAW_POSE       = True      # yes/no
    DRAW_RADAR      = True      # yes/no
    MATCH_DISTANCE  = 0.067     # seconds

    # CAMERA SETTINGS
    WEBCAM_DEVICE   = 0         # id of webcam
    FRAME_WIDTH     = 640       # pixels
    FRAME_HEIGHT    = 480       # pixels
    SAVE_RAW_IMG    = True      # yes/no

    # POSE SETTINGS
    POSE_CONFIDENCE = 0.6       # %

    # RADAR SETTINGS
    MIN_R = 100                 # cm
    MAX_R = 250                 # cm
    RES_R = 5                   # cm
    MIN_T = -30                 # deg
    MAX_T = 30                  # deg
    RES_T = 1                   # deg
    MIN_P = -30                 # deg
    MAX_P = 30                  # deg
    RES_P = 1                   # deg

class Config(object):
    # GENERAL SETTINGS
    LOGGING_RATE    = 1         # seconds
    DRAW_CAM        = False     # yes/no
    DRAW_POSE       = True      # yes/no
    DRAW_RADAR      = True      # yes/no

    # CAMERA SETTINGS
    FRAME_WIDTH     = 640       # pixels
    FRAME_HEIGHT    = 480       # pixels
    SAVE_RAW_IMG    = True      # yes/no

    # POSE SETTINGS
    POSE_CONFIDENCE = 0.6       # %

    # RADAR SETTINGS
    MIN_R = 100                 # cm
    MAX_R = 349                 # cm
    RES_R = 5                   # cm
    MIN_T = -44                 # deg
    MAX_T = 44                  # deg
    RES_T = 2.8                 # deg
    MIN_P = -44                 # deg
    MAX_P = 44                  # deg
    RES_P = 2.8                 # deg

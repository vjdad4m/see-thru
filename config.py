class Config(object):
    # GENERAL SETTINGS
    LOGGING_RATE    = 4         # seconds
    DRAW_CAM        = True      # yes/no
    DRAW_POSE       = True      # yes/no
    DRAW_RADAR      = True      # yes/no

    # CAMERA SETTINGS
    FRAME_WIDTH     = 640       # pixels
    FRAME_HEIGHT    = 480       # pixels
    SAVE_RAW_IMG    = True      # yes/no

    # POSE SETTINGS
    POSE_CONFIDENCE = 0.6       # %

    # RADAR SETTINGS
    MIN_R = 1                   # cm
    MAX_R = 240                 # cm
    RES_R = 4                   # cm
    MIN_T = -45                 # deg
    MAX_T = 45                  # deg
    RES_T = 5                   # deg
    MIN_P = -45                 # deg
    MAX_P = 45                  # deg
    RES_P = 5                   # deg

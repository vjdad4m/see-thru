class Config:
    radar_config = {
        'MinR': 100, 'MaxR': 250, 'ResR': 5,
        'MinT': -30, 'MaxT': 30,  'ResT': 1,
        'MinP': -30, 'MaxP': 30,  'ResP': 1 }
    
    draw_cam = True
    draw_radar = True
    
    match_distance = 0.067 # distance from measurements in seconds

    fill_in_missing_keypoints = True
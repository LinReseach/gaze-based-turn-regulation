import numpy as np
import time
import pandas as pd
import numpy as np
from numpy.linalg import inv
import numpy as np
from collections import deque
import time
from config import (
    SMOOTHING_WINDOW,
    READING_WINDOW_SIZE,
    SHIFT_WINDOW_SIZE,
    READING_THRESHOLD,
    SHIFT_THRESHOLD,
    RECT_CENTERS,
    RECT_WIDTHS,
    RECT_HEIGHTS
)



def smooth_gaze(gaze_data, new_pitch, new_yaw):
    """Apply moving average smoothing to pitch and yaw."""
    gaze_data.append((new_pitch, new_yaw))
    if len(gaze_data) < SMOOTHING_WINDOW:
        return new_pitch, new_yaw  # Return raw values until window is full
    smoothed_pitch = np.mean([p for p, y in gaze_data])
    smoothed_yaw = np.mean([y for p, y in gaze_data])
    return smoothed_pitch, smoothed_yaw


def analyze_reading_window(reading_window_data, threshold=READING_THRESHOLD):
    """Analyze the gaze data within the 3-second window to confirm reading."""
    if len(reading_window_data) < READING_WINDOW_SIZE:
        return False  # Not enough data to confirm reading

    # Count how many frames were looking at AOI1
    aoi1_count = sum(1 for _, aoi in reading_window_data if aoi == 'AOI 2')
    total_count = len(reading_window_data)
    aoi1_percentage = aoi1_count / total_count

    # Return True if the human is reading (looking at AOI1 for at least threshold%)
    return aoi1_percentage >= threshold


def analyze_shift_window(shift_window_data, threshold=SHIFT_THRESHOLD):
    """Analyze the gaze data within the 1-second window to detect gaze shift."""
    if len(shift_window_data) < SHIFT_WINDOW_SIZE:
        return False  # Not enough data to confirm shift

    # Count how many frames were looking outside AOI1
    #not_aoi1_count = sum(1 for _, aoi in shift_window_data if aoi != 'AOI 2')
    not_aoi1_count = sum(1 for _, aoi in shift_window_data if aoi == 'AOI 1')
    total_count = len(shift_window_data)
    not_aoi1_percentage = not_aoi1_count / total_count

    # Return True if the human has shifted gaze (looking outside AOI1 for at least threshold%)
    return not_aoi1_percentage >= threshold




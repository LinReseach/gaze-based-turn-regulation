import numpy as np
import time
import pandas as pd
import numpy as np
from numpy.linalg import inv
from config import (
    d_horizontal_robot_screen,
    d_vertical_robot_screen,
    RECT_CENTERS,
    RECT_WIDTHS,
    RECT_HEIGHTS,
    r_left,
    pos,
    h_eye_cam
)


def get_ladybug_to_eye_matrix(dir_eyes):
    """Creates a transformation matrix from the eye coordinate system."""
    up_vector = np.array([0, 0, 1], np.float32)
    z_axis = dir_eyes.flatten()
    x_axis = np.cross(up_vector, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    return np.stack([x_axis, y_axis, z_axis], axis=0)

def transform(g_p, pos, h_eye_cam, r_left):
    """Transforms gaze direction into the robot's coordinate system."""
    eye_pos = np.array([-100, -r_left, h_eye_cam]) if pos == 2 else np.array([0, 0, 0])  # Adjust as needed

    # Direction of eyes
    dir_eyes = eye_pos / np.linalg.norm(eye_pos)
   
    # Compute gaze coordinate system
    gaze_cs = get_ladybug_to_eye_matrix(dir_eyes)
    gaze_dir_lb = np.matmul(inv(gaze_cs), g_p.T)
   
    # Scaling factor for projection onto robot screen plane
    k = (d_horizontal_robot_screen - eye_pos[0]) / gaze_dir_lb[0]
    target = (gaze_dir_lb.T * k) + eye_pos
   
    # Adjust target coordinates for robot's screen plane
    target[:, 0] = d_horizontal_robot_screen - target[:, 0]
    target[:, 1] = -target[:, 1]
    target[:, 2] = target[:, 2] - d_vertical_robot_screen
   
    # Output as DataFrame with named columns
    return pd.DataFrame(target, columns=['virtual2d_x', 'virtual2d_y', 'depth'])

def find_aoi(x, y):
    # Loop through each AOI and check if the point is within the AOI's boundary
    for i, (center, width, height) in enumerate(zip(RECT_CENTERS, RECT_WIDTHS, RECT_HEIGHTS)):
        x_center, y_center = center
        half_width = width / 2
        half_height = height / 2

        # Define boundaries for the current AOI
        x_min = x_center - half_width
        x_max = x_center + half_width
        y_min = y_center - half_height
        y_max = y_center + half_height

        # Check if the point is within the boundaries
        if x_min <= x <= x_max and y_min <= y <= y_max:
            return f"AOI {i + 1}"

    # If the point is not in any AOI
    return "elsewhere"

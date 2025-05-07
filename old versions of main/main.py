# main.py

import argparse
import time
import numpy as np
from collections import deque
import cv2
import torch
from config import (
    # DEFAULT_IP, DEFAULT_PORT, DEFAULT_CAMERA_ID,
    # DEFAULT_SNAPSHOT_PATH, DEFAULT_MODEL_ARCH,
    FRAME_RATE
)
#
from pepper_connection import SocketConnection
from gaze_prediction import getArch, prediction, get_transformations
from state_detection import (
    smooth_gaze, analyze_reading_window,
    analyze_shift_window
)
from aoi import get_ladybug_to_eye_matrix,transform,find_aoi


def parse_args():
    """Parse input arguments."""

    """model parameters"""
    parser = argparse.ArgumentParser(
        description='Gaze evalution using model pretrained with L2CS-Net on Gaze360.')
    parser.add_argument(
        '--gpu',dest='gpu_id', help='GPU device id to use [0]',
        default="0", type=str)
    parser.add_argument(
        '--snapshot',dest='snapshot', help='Path of model snapshot.',
        default='output/snapshots/L2CS-gaze360-_loader-180-4/_epoch_55.pkl', type=str)
    parser.add_argument(
        '--arch',dest='arch',help='Network architecture, can be: ResNet18, ResNet34, ResNet50, ResNet101, ResNet152',
        default='ResNet50', type=str)

    """connection parameters"""
    parser.add_argument("--ip", type=str, default="127.0.0.1",
                        help="Robot IP address. Default 127.0.0.1")
    parser.add_argument("--port", type=int, default=12345,
                        help="Pepper port number. Default 9559.")
    parser.add_argument("--cam_id", type=int, default=3,
                        help="Camera id according to pepper docs. Use 3 for "
                             "stereo camera and 0. Default is 3.")
    args = parser.parse_args()

    return args


def main():
    # Parse arguments
    args = parse_args()

    # Create socket connection with Pepper
    connect = SocketConnection(ip=args.ip, port=args.port, camera=args.cam_id)

    # Prepare video recording
    video_name = 'pepper_example.avi'
    frame = connect.get_img()
    cv2.imshow('img', frame)
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_name, 0, 5, (width, height))

    # Prepare model and transformations
    transformations = get_transformations()
    model = getArch(args.arch, 90)
    
    # Load model snapshot
    saved_state_dict = torch.load(args.snapshot)
    model.load_state_dict(saved_state_dict, strict=False)

    # Initialize data collections
    gaze_data = deque(maxlen=10)
    reading_window_data = deque(maxlen=30)
    shift_window_data = deque(maxlen=10)

    # Tracking variables
    is_reading = False
    flag = 0
    url = 'http://ibb.co/LXYDddT6'

    # Begin interaction loop
    connect.say('Please complete the task')
    time.sleep(1)
    connect.tablet(url)
    time.sleep(1)
    while True:
        start_time = time.time()
        frame = connect.get_img()
        img, pitch, yaw = prediction(transformations, model, frame)
        print(pitch, yaw)
        if (pitch, yaw) != (None, None):
            smoothed_pitch, smoothed_yaw = smooth_gaze(gaze_data, pitch, yaw)

            # Convert gaze direction to Cartesian coordinates
            x = np.cos(smoothed_pitch) * np.sin(smoothed_yaw)
            y = np.sin(smoothed_pitch)
            z = -np.cos(smoothed_yaw) * np.cos(smoothed_pitch)
            g_p = np.array([[x, y, z]])

            # Transform gaze to robot's coordinate system
            dfn = transform(g_p, pos=2, h_eye_cam=1.2, r_left=0)[['virtual2d_x', 'virtual2d_y', 'depth']]

            output_x = dfn['virtual2d_y'].values[0]
            output_y = dfn['depth'].values[0]

            current_aoi = find_aoi(output_x, output_y)
            print(current_aoi)
            current_time = time.time()

            reading_window_data.append((current_time, current_aoi))
            shift_window_data.append((current_time, current_aoi))

            # Reading detection logic
            if not is_reading:
                if analyze_reading_window(reading_window_data):
                    is_reading = True
                    flag = 1
            else:
                if analyze_shift_window(shift_window_data) and flag == 1:
                    is_reading = False
                    connect.say('It seems you finished the task, let us go next.')
                    break

        # Maintain consistent frame rate
        elapsed_time = time.time() - start_time
        sleep_time = max(0, (1.0 / FRAME_RATE) - elapsed_time)
        time.sleep(sleep_time)

if __name__ == '__main__':
    main()



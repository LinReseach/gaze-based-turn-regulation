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
    FRAME_RATE,url_list,text_list,pos, h_eye_cam, r_left,
    SMOOTHING_WINDOW, READING_WINDOW_SIZE,SHIFT_WINDOW_SIZE


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
    #video_name = 'pepper_example.avi'
    frame = connect.get_img()
    cv2.imshow('img', frame)
    height, width, layers = frame.shape
    #video = cv2.VideoWriter(video_name, 0, 5, (width, height))

    # Prepare model and transformations
    transformations = get_transformations()
    model = getArch(args.arch, 90)
    
    # Load model snapshot
    saved_state_dict = torch.load(args.snapshot)
    model.load_state_dict(saved_state_dict, strict=False)

    # Initialize data collections
    gaze_data = deque(maxlen=SMOOTHING_WINDOW)
    reading_window_data = deque(maxlen=READING_WINDOW_SIZE)
    shift_window_data = deque(maxlen=SHIFT_WINDOW_SIZE)

    # Tracking variables


    # Initialize index
    # Initialize index
    i = 0

    # Begin interaction loop
    while i < len(text_list):
        # Display text and tablet content
        connect.say(text_list[i])
        time.sleep(2)
        connect.tablet(url_list[i])
        time.sleep(1)

        # Reset reading state for new content
        is_reading = False
        reading_detected = False
        shift_detected = False

        # Clear data collections for new content
        gaze_data.clear()
        reading_window_data.clear()
        shift_window_data.clear()

        print(f"Showing content {i + 1}/{len(text_list)}: {text_list[i]}")

        while True:
            start_time = time.time()
            frame = connect.get_img()
            img, pitch, yaw = prediction(transformations, model, frame)
            print(f"Raw gaze: pitch={pitch}, yaw={yaw}")

            if (pitch, yaw) != (None, None):
                smoothed_pitch, smoothed_yaw = smooth_gaze(gaze_data, pitch, yaw)
                print(f"Smoothed gaze: pitch={smoothed_pitch}, yaw={smoothed_yaw}")

                # Convert gaze direction to Cartesian coordinates
                x = np.cos(smoothed_pitch) * np.sin(smoothed_yaw)
                y = np.sin(smoothed_pitch)
                z = -np.cos(smoothed_yaw) * np.cos(smoothed_pitch)
                g_p = np.array([[x, y, z]])

                # Transform gaze to robot's coordinate system
                dfn = transform(g_p, pos, h_eye_cam, r_left)[['virtual2d_x', 'virtual2d_y', 'depth']]

                output_x = dfn['virtual2d_y'].values[0]
                output_y = dfn['depth'].values[0]

                current_aoi = find_aoi(output_x, output_y)
                current_time = time.time()
                print(f"Current AOI: {current_aoi}")

                reading_window_data.append((current_time, current_aoi))
                shift_window_data.append((current_time, current_aoi))

                # Reading detection logic
                if not is_reading:
                    reading_detected = analyze_reading_window(reading_window_data)
                    if reading_detected:
                        is_reading = True
                        print(f"Reading detected for content {i + 1}")
                else:
                    shift_detected = analyze_shift_window(shift_window_data)
                    if shift_detected:
                        print(f"Shift detected after reading content {i + 1}, moving to next content")
                        i += 1  # Move to the next step in the interaction
                        break

            # Maintain consistent frame rate
            elapsed_time = time.time() - start_time
            sleep_time = max(0, (1.0 / FRAME_RATE) - elapsed_time)
            time.sleep(sleep_time)
            print(sleep_time)

if __name__ == '__main__':
    main()



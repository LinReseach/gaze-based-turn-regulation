# main.py
import argparse
import time
import numpy as np
from collections import deque
import cv2
import torch
import os
import csv
from config import (
    # DEFAULT_IP, DEFAULT_PORT, DEFAULT_CAMERA_ID,
    # DEFAULT_SNAPSHOT_PATH, DEFAULT_MODEL_ARCH,
    FRAME_RATE,url_list,text_list,pos, h_eye_cam, r_left,
    SMOOTHING_WINDOW, READING_WINDOW_SIZE,SHIFT_WINDOW_SIZE,
    participant_num, scenario_num
)
#
from pepper_connection import SocketConnection
from gaze_prediction import getArch, prediction, get_transformations
from state_detection import (
    smooth_gaze, analyze_reading_window,
    analyze_shift_window
)
from aoi import get_ladybug_to_eye_matrix,transform,find_aoi
from threading import Thread



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
    time.sleep(1)
    connect.adjust_head(-0.2, 0)
    time.sleep(1)

    # Prepare video recording
    video_name = 'pepper_example.avi'
    frame = connect.get_img()
    #cv2.imshow('img', frame)
    height, width, layers = frame.shape
    #video = cv2.VideoWriter(video_name, 0, 5, (width, height))

    # Create output directories if they don't exist
    filename='calibration_par'+str(participant_num)
    os.makedirs(filename, exist_ok=True)
    images_dir = filename+'/calibration_images'
    os.makedirs(images_dir, exist_ok=True)
    
    # Setup CSV file for AOI tracking
    csv_file = filename+'/calibration.csv'
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame_id', 'image_filename', 'timestamp', 'aoi', 'pitch', 'yaw',
                         'smoothed_pitch', 'smoothed_yaw','is_reading'])

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

    url = 'http://ibb.co/LXYDddT6'
    
    # Frame counter for image naming

  
    # Begin interaction loop


    def recording_loop():
        is_reading = False
        flag = 0
        frame_counter = 0
        while True:
            start_time = time.time()
            frame = connect.get_img()

            # Increment frame counter
            frame_counter += 1

            # Generate image filename with leading zeros
            img_filename = f"{frame_counter:04d}.jpg"
            img_path = os.path.join(images_dir, img_filename)

            # Save the current frame
            #cv2.imwrite(img_path, frame)

            # Process image for gaze prediction
            img, pitch, yaw = prediction(transformations, model, frame)

            cv2.imwrite(img_path, frame)
            #cv2.imshow('calibration_img',img)

            # Current timestamp
            current_time = time.time()

            # Default AOI value when pitch/yaw detection fails
            current_aoi = None

            if (pitch, yaw) != (None, None):
                smoothed_pitch, smoothed_yaw = smooth_gaze(gaze_data, pitch, yaw)

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
                print(f"Frame {frame_counter}: AOI = {current_aoi}, Pitch = {pitch:.2f}, Yaw = {yaw:.2f}")

                reading_window_data.append((current_time, current_aoi))
                shift_window_data.append((current_time, current_aoi))

                # Reading detection logic
                if not is_reading:
                    if analyze_reading_window(reading_window_data):
                        is_reading = True
                        flag = 1
                        print("Reading detected!")
                else:
                    if analyze_shift_window(shift_window_data) and flag == 1:
                        is_reading = False
                        print("Reading completed, moving to next task")
                        connect.say('It seems you finished the task, let us go next.')
                        break
            else:
                print(f"Frame {frame_counter}: No gaze detected")

            # Write data to CSV
            with open(csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    frame_counter,
                    img_filename,
                    current_time,
                    current_aoi if current_aoi is not None else "None",
                    pitch if pitch is not None else "None",
                    yaw if yaw is not None else "None",
                    smoothed_pitch if pitch is not None else "None",
                    smoothed_yaw if yaw is not None else "None",
                    is_reading
                ])

            # Maintain consistent frame rate
            elapsed_time = time.time() - start_time
            sleep_time = max(0, (1.0 / FRAME_RATE) - elapsed_time)
            time.sleep(sleep_time)

    Thread(target=recording_loop).start()
    connect.say('Please look at my head camera')
    time.sleep(5)
    connect.say('now, look at my tablet')
    time.sleep(2)
    connect.tablet(url)
    time.sleep(5)
    connect.say('Calibration complete.')



if __name__ == '__main__':
    main()



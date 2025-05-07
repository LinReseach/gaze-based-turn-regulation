# main.py
# look at face to continue:
# (1)state_detection.py -not_aoi1_count = sum(1 for _, aoi in shift_window_data if aoi == 'AOI 1')
# (2)config.py-SHIFT_THRESHOLD = 0.4
import argparse
import time
import numpy as np
import os
import csv
from collections import deque
import cv2
import torch
from datetime import datetime
from config import (
    # DEFAULT_IP, DEFAULT_PORT, DEFAULT_CAMERA_ID,
    # DEFAULT_SNAPSHOT_PATH, DEFAULT_MODEL_ARCH,
    FRAME_RATE, url_list, text_list, pos, h_eye_cam, r_left,
    SMOOTHING_WINDOW, READING_WINDOW_SIZE, SHIFT_WINDOW_SIZE,
    participant_num, scenario_num
)
#
from pepper_connection import SocketConnection
from gaze_prediction import getArch, prediction, get_transformations
from state_detection import (
    smooth_gaze, analyze_reading_window,
    analyze_shift_window
)
from aoi import get_ladybug_to_eye_matrix, transform, find_aoi


def parse_args():
    """Parse input arguments."""

    """model parameters"""
    parser = argparse.ArgumentParser(
        description='Gaze evalution using model pretrained with L2CS-Net on Gaze360.')
    parser.add_argument(
        '--gpu', dest='gpu_id', help='GPU device id to use [0]',
        default="0", type=str)
    parser.add_argument(
        '--snapshot', dest='snapshot', help='Path of model snapshot.',
        default='output/snapshots/L2CS-gaze360-_loader-180-4/_epoch_55.pkl', type=str)
    parser.add_argument(
        '--arch', dest='arch', help='Network architecture, can be: ResNet18, ResNet34, ResNet50, ResNet101, ResNet152',
        default='ResNet50', type=str)

    """connection parameters"""
    parser.add_argument("--ip", type=str, default="127.0.0.1",
                        help="Robot IP address. Default 127.0.0.1")
    parser.add_argument("--port", type=int, default=12345,
                        help="Pepper port number. Default 9559.")
    parser.add_argument("--cam_id", type=int, default=3,
                        help="Camera id according to pepper docs. Use 3 for "
                             "stereo camera and 0. Default is 3.")
    parser.add_argument("--output_dir", type=str, default="experiment_data_par"+str(participant_num)+"sc"+str(scenario_num),
                        help="Directory to save frames and CSV data")
    args = parser.parse_args()

    return args


def setup_output_directory(output_dir):
    """Create output directory structure for saving frames and CSV"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    frames_dir = os.path.join(output_dir, "frames")
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)
        
    return frames_dir


def save_frame(frame, frame_count, frames_dir):
    """Save the current frame as a numbered image"""
    frame_filename = os.path.join(frames_dir, f"{frame_count:05d}.jpg")
    cv2.imwrite(frame_filename, frame)
    return frame_filename


def main():
    # Parse arguments
    args = parse_args()
    
    # Setup output directory for saving data
    frames_dir = setup_output_directory(args.output_dir)
    csv_path = os.path.join(args.output_dir, "gaze_data.csv")
    
    # Initialize CSV file with headers
    with open(csv_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['frame_name', 'timestamp', 'aoi', 'pitch', 'yaw', 'reading_state', 'page_number'])

    # Create socket connection with Pepper
    connect = SocketConnection(ip=args.ip, port=args.port, camera=args.cam_id)
    time.sleep(1)
    connect.adjust_head(-0.2, 0)
    time.sleep(1)
    connect.idle()
    time.sleep(2)

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

    # Initialize frame counter
    frame_count = 1
    
    # Record experiment timing
    experiment_start_time = None
    experiment_end_time = None
    
    # Initialize index
    i = 1
    # time.sleep(1)
    connect.tablet(url_list[0])
    time.sleep(1)  # 1
    connect.say(text_list[0])
    time.sleep(17)
    
    # Begin interaction loop
    while i < len(text_list):
        # Display text and tablet content
        time.sleep(1)
        connect.say(text_list[i])
        
        if i == 7:
            time.sleep(2.5)
        else:
            time.sleep(2)
            
        connect.tablet(url_list[i])
        time.sleep(1)
        
        # Record start time for first tablet content
        if i == 1 and experiment_start_time is None:
            experiment_start_time = time.time()

        # Reset reading state for new content
        is_reading = False
        reading_detected = False
        shift_detected = False

        # Clear data collections for new content
        gaze_data.clear()
        reading_window_data.clear()
        shift_window_data.clear()

        print(f"Showing content {i + 1}/{len(text_list)}: {text_list[i]}")
        start_page = time.time()
        
        while True:
            start_time = time.time()
            frame = connect.get_img()
            
            # Save the frame
            frame_filename = save_frame(frame, frame_count, frames_dir)
            frame_count += 1
            
            img, pitch, yaw = prediction(transformations, model, frame)
            print(f"Raw gaze: pitch={pitch}, yaw={yaw}")
            
            # Default values for CSV
            current_aoi = "None"
            smoothed_pitch = None
            smoothed_yaw = None
            reading_state = "not_reading"
            
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

                # Update reading state
                if is_reading:
                    reading_state = "reading"
                
                # Reading detection logic
                if not is_reading:
                    reading_detected = analyze_reading_window(reading_window_data)
                    if reading_detected:
                        is_reading = True
                        reading_state = "reading"
                        print(f"Reading detected for content {i + 1}")
                else:
                    shift_detected = analyze_shift_window(shift_window_data)
                    if shift_detected:
                        print(f"Shift detected after reading content {i + 1}, moving to next content")
                        
                        # If this is the last page, record experiment end time
                        if i == len(text_list) - 1:
                            experiment_end_time = time.time()
                        
                        i += 1  # Move to the next step in the interaction
                        end_page = time.time()
                        print(f"Time spent on page: {end_page - start_page}, Total gaze datapoints: {len(gaze_data)}")
                        break
            
            # Save data to CSV
            frame_name = os.path.basename(frame_filename)
            with open(csv_path, 'a', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow([
                    frame_name,
                    time.time(),
                    current_aoi,
                    smoothed_pitch if smoothed_pitch is not None else pitch,
                    smoothed_yaw if smoothed_yaw is not None else yaw,
                    reading_state,
                    i
                ])

            # Maintain consistent frame rate
            elapsed_time = time.time() - start_time
            sleep_time = max(0, (1.0 / FRAME_RATE) - elapsed_time)
            time.sleep(sleep_time)
            print(f"elapsed_time: {elapsed_time}")
    
    # Print total experiment duration
    if experiment_start_time and experiment_end_time:
        total_duration = experiment_end_time - experiment_start_time
        print(f"Total experiment duration from url_list[1] to url_list[{len(url_list)-1}]: {total_duration:.2f} seconds")
        
        # Save the experiment summary
        with open(os.path.join(args.output_dir, "experiment_summary.txt"), 'w') as f:
            f.write(f"Experiment Start Time: {datetime.fromtimestamp(experiment_start_time)}\n")
            f.write(f"Experiment End Time: {datetime.fromtimestamp(experiment_end_time)}\n")
            f.write(f"Total Duration: {total_duration:.2f} seconds\n")
            f.write(f"Total Frames Captured: {frame_count-1}\n")

if __name__ == '__main__':
    main()

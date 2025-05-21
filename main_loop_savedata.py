# main.py - Modified to save frames and log data
import argparse
import time
import numpy as np
from collections import deque
import cv2
import torch
import os
import csv
from datetime import datetime
from config import (
    FRAME_RATE, url_list, text_list, pos, h_eye_cam, r_left,
    SMOOTHING_WINDOW, READING_WINDOW_SIZE, SHIFT_WINDOW_SIZE
)

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
    parser.add_argument("--output_dir", type=str, default="experiment_data",
                        help="Directory to save frames and data. Default 'experiment_data'")
    args = parser.parse_args()

    return args


def setup_output_directories(output_dir):
    """Create directories for saving frames and data."""
    # Create main output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create frames directory
    frames_dir = os.path.join(output_dir, "frames")
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)
    
    return frames_dir


def save_frame(frame, frame_count, frames_dir):
    """Save a frame as a numbered image."""
    frame_filename = os.path.join(frames_dir, f"{frame_count:05d}.jpg")
    cv2.imwrite(frame_filename, frame)
    return frame_filename


def initialize_csv(output_dir):
    """Initialize CSV file for logging gaze data."""
    csv_filename = os.path.join(output_dir, f"gaze_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['frame_name', 'timestamp', 'aoi', 'pitch', 'yaw', 'reading_state', 'page_number'])
    
    return csv_filename


def log_gaze_data(csv_filename, frame_name, timestamp, aoi, pitch, yaw, reading_state, page_number):
    """Log gaze data to CSV file."""
    with open(csv_filename, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([frame_name, timestamp, aoi, pitch, yaw, reading_state, page_number])


def main():
    # Parse arguments
    args = parse_args()

    # Setup output directories
    frames_dir = setup_output_directories(args.output_dir)
    csv_filename = initialize_csv(args.output_dir)
    
    # Initialize timing log for page transitions
    timing_log = os.path.join(args.output_dir, "page_timing.csv")
    with open(timing_log, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['page_number', 'url', 'start_time', 'end_time', 'duration'])

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
    frame_count = 0
    experiment_start_time = time.time()

    # Initialize index
    i = 1
    # time.sleep(1)
    connect.tablet(url_list[0])
    start_time_first_page = time.time()
    time.sleep(1)  # 1
    connect.say(text_list[0])
    time.sleep(17)
    
    # Record timing for introduction page
    with open(timing_log, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([0, url_list[0], start_time_first_page, time.time(), time.time() - start_time_first_page])
    
    # Begin interaction loop
    while i < len(text_list):
        # Display text and tablet content
        time.sleep(1)
        connect.say(text_list[i])
        
        if i == 7:
            time.sleep(2.5)
        else:
            time.sleep(2)
            
        page_start_time = time.time()
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
        start_page = time.time()
        
        while True:
            loop_start_time = time.time()
            frame = connect.get_img()
            
            # Save the frame
            frame_count += 1
            frame_name = f"{frame_count:05d}.jpg"
            frame_path = save_frame(frame, frame_count, frames_dir)
            
            img, pitch, yaw = prediction(transformations, model, frame)
            print(f"Raw gaze: pitch={pitch}, yaw={yaw}")
            
            # Default values in case gaze tracking fails
            current_aoi = "None"
            reading_state = "Unknown"
            
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
                        reading_state = "Reading"
                        print(f"Reading detected for content {i + 1}")
                    else:
                        reading_state = "Not Reading"
                else:
                    shift_detected = analyze_shift_window(shift_window_data)
                    reading_state = "Reading"
                    if shift_detected:
                        print(f"Shift detected after reading content {i + 1}, moving to next content")
                        reading_state = "Shifted"
                        
                        # Record timing for completed page
                        page_end_time = time.time()
                        page_duration = page_end_time - page_start_time
                        with open(timing_log, 'a', newline='') as csvfile:
                            csv_writer = csv.writer(csvfile)
                            csv_writer.writerow([i, url_list[i], page_start_time, page_end_time, page_duration])
                        
                        i += 1  # Move to the next step in the interaction
                        break
            
            # Log the gaze data
            log_gaze_data(csv_filename, frame_name, time.time(), current_aoi,
                          pitch if pitch is not None else "None",
                          yaw if yaw is not None else "None",
                          reading_state, i)

            # Maintain consistent frame rate
            elapsed_time = time.time() - loop_start_time
            sleep_time = max(0, (1.0 / FRAME_RATE) - elapsed_time)
            time.sleep(sleep_time)
            print(f"Frame processing time: {elapsed_time:.4f}s, Sleep time: {sleep_time:.4f}s")
    
    # Record total experiment duration
    experiment_end_time = time.time()
    total_duration = experiment_end_time - experiment_start_time
    
    # Log overall timing
    with open(os.path.join(args.output_dir, "experiment_summary.txt"), 'w') as f:
        f.write(f"Experiment started at: {datetime.fromtimestamp(experiment_start_time)}\n")
        f.write(f"Experiment ended at: {datetime.fromtimestamp(experiment_end_time)}\n")
        f.write(f"Total duration: {total_duration:.2f} seconds\n")
        f.write(f"Total frames captured: {frame_count}\n")
        f.write(f"Average frame rate: {frame_count/total_duration:.2f} fps\n")

    print(f"Experiment completed. Total duration: {total_duration:.2f} seconds")
    print(f"Data saved to {args.output_dir}")

if __name__ == '__main__':
    main()

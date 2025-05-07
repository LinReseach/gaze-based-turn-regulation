
import os
import pandas as pd
import numpy as np

participant_num=0
scenario_num=1

eye_height = 130  # Example value for eye height
camera_height = 120
h_eye_cam = eye_height - camera_height
# Parameters for transformation
d_horizontal_robot_screen = 0
d_vertical_robot_screen = 0
r_left = 0  # Adjust as needed
pos = 2  # Example position input; adjust as necessary
# Define AOI centers, widths, and heights
RECT_CENTERS = [(0, -8.1),(0,-28),(-30.8,-32), (29.7,-30.8)]#tablet_original(0,-32)
RECT_WIDTHS = [80 * 2 / 9.19+2, 24.6+5, 150 * 2 / 9.19, 150 * 2 / 9.19]
RECT_HEIGHTS = [80 * 2 / 9.19+2, 17.5+5, 150 * 2 / 9.19, 150 * 2 / 9.19]
#pre-definition
# Parameters for smoothing and time window analysis
SMOOTHING_WINDOW = 3 # Number of frames for moving average smoothing (increased for low precision)
READING_TIME_WINDOW = 0.5 # 1s  3-second window to confirm sustained gaze at AOI1
SHIFT_TIME_WINDOW = 0.5 # 0.5s 1-second window to detect gaze shift
FRAME_RATE = 8 #8 Frametime.sleep(1) rate (given as 10 fps)
READING_WINDOW_SIZE = int(READING_TIME_WINDOW * FRAME_RATE)  # Frames in 3-second window
SHIFT_WINDOW_SIZE = int(SHIFT_TIME_WINDOW * FRAME_RATE)  # Frames in 1-second window
READING_THRESHOLD = 0.4  # Require 30% of frames in AOI1 to confirm reading (increased for robustness)
SHIFT_THRESHOLD = 0.5#0.7 # Require 70% of frames outside AOI1 to confirm shift
url_list = ['https://iili.io/3XGZDDG.png',
            'https://ibb.co/tfXn58v', 'https://ibb.co/LDSQbhLQ', 'https://ibb.co/wZXYFRMB', 'https://ibb.co/mV8Z1VcQ',
            'https://ibb.co/0yjZJQ2V', 'https://ibb.co/Xrkp57kN', 'https://ibb.co/vC8gssGj']



# #text_list = ['Please complete the task. When you finish reading, look at me to continue', 'Now, take a look at their picture',
#              'Here is some information about one colleague role', 'Here’s some information about another colleague role',
#              'This is where one colleague work.','This is where another colleague work.','done']

text_list= [
    "Hey there! I am Pepper. Really nice to meet you. First day, huh? Don't worry, I'll help you get the hang of things. I've got some important stuff to show you on my tablet. Just give me a signal when you're done reading each section and I'll guide you to what's next.",
    "Alright, get ready!",
    "Here are their names and photos.",
    "Now, here’s  what the first colleague does.",
    "this is what the second colleague does.",
    "Here’s the location for the first colleague.",
    "And this is where the second colleague’s office is.",
    "Great job! Now, let’s see how much you remember."
]





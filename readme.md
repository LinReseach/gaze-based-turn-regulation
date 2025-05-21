# Gaze-Contingent Robot Interaction System

This repository contains the implementation of a gaze-contingent interaction system for the Pepper robot, enabling natural turn-taking through eye gaze detection. The system detects when a user is reading content on the robot's tablet and automatically advances to the next information screen when the user looks at the robot's face, creating a more intuitive and seamless interaction experience.

## Overview

The interaction flow works as follows:
1. The robot displays content on its tablet and verbally introduces it
2. The system tracks the user's gaze to detect when they are reading the tablet content
3. When the user finishes reading and looks at the robot's face, the system detects this gaze shift
4. The robot automatically advances to the next piece of content, creating a natural conversational flow

This implementation provides a novel example of embodied conversational turn-taking mediated through gaze, allowing for more human-like interactions without requiring explicit verbal commands or button presses.

## Features

- Real-time gaze tracking using L2CS-Net with a pre-trained model
- Gaze coordinate transformation to the robot's reference frame
- Area of Interest (AOI) definition and detection
- Smoothed gaze analysis with time-window based state detection
- Automatic content progression based on gaze patterns
- Complete interaction flow management
- Optional data collection mode for research purposes



## Hardware Setup

- Pepper robot 
- Network connection between the robot and the computer running this codes
- GPU is recommended for real-time gaze detection performance(we use GeForce RTX 3070 8GB)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/gaze-contingent-robot-interaction.git
cd gaze-contingent-robot-interaction
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Download the pre-trained L2CS-Net model:
```bash
mkdir -p models/Gaze360-20220914T091057Z-001/Gaze360/L2CSNet_gaze360.pkl
# Download the model to the above directory(for example)
```

## Configuration

The system behavior can be customized in `config.py`:

- `RECT_CENTERS`, `RECT_WIDTHS`, `RECT_HEIGHTS`: Define the Areas of Interest (AOIs)
- `SMOOTHING_WINDOW`: Number of frames for moving average smoothing
- `READING_TIME_WINDOW`, `SHIFT_TIME_WINDOW`: Time windows for detecting reading and gaze shift
- `READING_THRESHOLD`, `SHIFT_THRESHOLD`: Thresholds for state detection
- `text_list`: Content to be displayed during the interaction
- `url_list`: URLs of the visual content for the tablet

## Usage

### Main Interaction Loop

To run the main interaction system:

```bash
python main_loop.py --ip <robot_ip> --port <port> --cam_id <camera_id> --snapshot <path_to_model> --arch ResNet50
```

Arguments:
- `--ip`: Robot's IP address (default: 127.0.0.1)
- `--port`: Port for communication (default: 12345)
- `--cam_id`: Camera ID (default: 3, use 3 for stereo camera)
- `--snapshot`: Path to the pre-trained model
- `--arch`: Network architecture (default: ResNet50)

### Data Collection Mode

For research purposes, you can run a data collection mode that saves frames and gaze data:

```bash
python main_check.py --ip <robot_ip>  --cam_id <camera_id> --snapshot <path_to_model> --port <port>

python3 main_loop_savedata_lightweight.py --ip=10.0.0.180  --cam_id=4 --snapshot models/Gaze360-20220914T091057Z-001/Gaze360/L2CSNet_gaze360.pkl --port 12348
```



## Code Structure

run in the computer
- `main_loop_savedata_lightweight.py`: Main interaction loop for the robot, save data usinhg a lightweight way
- `main_loop_savedata_lightweight_v2.py`: Main interaction loop for the robot, save data usinhg a lightweight way, run `gaze_prediction_v2.py`
- `config.py`: Configuration parameters
- `aoi.py`: Area of Interest detection and coordinate transformations
- `gaze_prediction.py`: Gaze prediction using L2CS-Net
- `gaze_prediction_v2.py`: Gaze prediction using L2CS-Net, drawing gaze vectors before and after average smoothing, and put text of aoi, state detection.
- `state_detection.py`: state detection logic
- `pepper_connection.py`: Communication with the Pepper robot
- `model.py`: L2CS neural network model definition
- `utils.py`: Utility functions for gaze visualization and calculations
- `requirements.txt`: for creating l2cs environment
- `camera4k.py`: record frames using 4k,(optional, this did not work somehow)
- `haarcascade_eye.xml`: for camera4k.py
- `haarcascade_frontalface_default.xml`:  for camera4k.py


run in the pepper robot 
 `server_tablet.py`: Communication with the computer

## How It Works

The system implements gaze-contingent turn-taking through these key components:

1. **Gaze Detection**: Uses a pre-trained L2CS-Net model to predict gaze direction from face images.

2. **Coordinate Transformation**: Transforms raw gaze predictions to the robot's coordinate system.

3. **AOI Detection**: Maps transformed gaze coordinates to defined Areas of Interest (tablet, robot face).

4. **State Detection**: 
   - **Reading Detection**: Analyzes if the user has been looking at the tablet content (AOI 2) for a sufficient time period.
   - **Shift Detection**: Detects when the user shifts their gaze from the tablet to the robot's face (AOI 1).

5. **Turn Management**: Advances content when the system detects the user has finished reading (looked at the tablet and then shifted gaze to the robot).

## Areas of Interest (AOIs)

The system defines these main AOIs:
- **AOI 1**: Robot's face area - used to detect when the user is looking at the robot
- **AOI 2**: Tablet content area - used to detect when the user is reading content
- **AOI 3 & 4**: Additional areas that can be used for more complex interactions

## Research Applications

This system can be used to study:
- Natural turn-taking patterns in human-robot interaction
- Gaze behavior during information consumption
- Implicit communication through eye gaze
- User experience with gaze-contingent interfaces

## License

[MIT License](LICENSE)

## Citation

If you use this code in your research, please cite:

```
@article{gazecontingentinteraction,
  title={Gaze-Contingent Turn-Taking in Human-Robot Interaction},
  author={Your Name},
  journal={Your Journal},
  year={2025}
}
```

## Acknowledgements

- The gaze estimation is based on L2CS-Net
- This project was developed as part of research on natural human-robot interaction patterns

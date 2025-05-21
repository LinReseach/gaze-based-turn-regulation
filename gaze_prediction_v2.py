import argparse
import random

import numpy as np
import cv2
import time
import socket
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision

import pandas as pd
import numpy as np

from PIL import Image
from utils import select_device, draw_gaze
from PIL import Image, ImageOps

from face_detection import RetinaFace
from model import L2CS

from numpy.linalg import inv







def getArch(arch,bins):
    # Base network structure
    if arch == 'ResNet18':
        model = L2CS( torchvision.models.resnet.BasicBlock,[2, 2,  2, 2], bins)
    elif arch == 'ResNet34':
        model = L2CS( torchvision.models.resnet.BasicBlock,[3, 4,  6, 3], bins)
    elif arch == 'ResNet101':
        model = L2CS( torchvision.models.resnet.Bottleneck,[3, 4, 23, 3], bins)
    elif arch == 'ResNet152':
        model = L2CS( torchvision.models.resnet.Bottleneck,[3, 8, 36, 3], bins)
    else:
        if arch != 'ResNet50':
            print('Invalid value for architecture is passed! '
                'The default value of ResNet50 will be used instead!')
        model = L2CS( torchvision.models.resnet.Bottleneck, [3, 4, 6,  3], bins)
    return model


def prediction(transformations, model, frame):
    cudnn.enabled = True

    # arch = args.arch
    batch_size = 16
    #cam = args.cam_id
    gpu = select_device("0", batch_size=batch_size)
    # snapshot_path = args.snapshot

    model.cuda(gpu)
    model.eval()

    softmax = nn.Softmax(dim=1)
    detector = RetinaFace(gpu_id=0)  # 0 for gpu, -1 for CPU
    idx_tensor = [idx for idx in range(90)]
    idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)
    x = 0

    pitch_predicted = None
    yaw_predicted = None

    start_fps = time.time()

    faces = detector(frame)
    face_bbox = None
    
    if faces is not None:
        for box, landmarks, score in faces:
            if score < .95:
                continue
            x_min = int(box[0])
            if x_min < 0:
                x_min = 0
            y_min = int(box[1])
            if y_min < 0:
                y_min = 0
            x_max = int(box[2])
            y_max = int(box[3])
            bbox_width = x_max - x_min
            bbox_height = y_max - y_min

            face_bbox = (x_min, y_min, bbox_width, bbox_height)
            
            # Crop image
            img = frame[y_min:y_max, x_min:x_max]
            img = cv2.resize(img, (224, 224))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            im_pil = Image.fromarray(img)
            img = transformations(im_pil)
            img = Variable(img).cuda(gpu)
            img = img.unsqueeze(0)

            # gaze prediction
            gaze_yaw, gaze_pitch = model(img)

            pitch_predicted = softmax(gaze_pitch)
            yaw_predicted = softmax(gaze_yaw)

            # Get continuous predictions in degrees.
            pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 4 - 180
            yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 4 - 180

            pitch_predicted = pitch_predicted.cpu().detach().numpy() * np.pi / 180.0
            yaw_predicted = yaw_predicted.cpu().detach().numpy() * np.pi / 180.0

            # Draw face bounding box
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
            
        myFPS = 1.0 / (time.time() - start_fps)
        #cv2.putText(frame, 'FPS: {:.1f}'.format(myFPS), (10, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1,
                   # cv2.LINE_AA)

    return frame, pitch_predicted, yaw_predicted, face_bbox

def get_aoi_name(aoi):
    if aoi == 'AOI 1':
        return "face"
    elif aoi == 'AOI 2':
        return "tablet"
    else:
        return "elsewhere"

# In your main processing loop
# Assuming 'aoi' variable already contains the current AOI value

    
def draw_annotated_frame(frame, face_bbox, original_gaze, smoothed_gaze, aoi, reading_state, page_number):
    """
    Draw both original and smoothed gaze vectors on the frame along with AOI, reading state, and page number
    
    Args:
        frame: The input frame
        face_bbox: Tuple of (x_min, y_min, width, height) for face detection
        original_gaze: Tuple of (pitch, yaw) for original gaze
        smoothed_gaze: Tuple of (pitch, yaw) for smoothed gaze
        aoi: Current area of interest
        reading_state: Current reading state ("reading" or "not_reading")
        page_number: Current page number
    
    Returns:
        Annotated frame
    """
    # Draw original gaze with red color if available
    if face_bbox is not None and original_gaze[0] is not None and original_gaze[1] is not None:
        draw_gaze(face_bbox[0], face_bbox[1], face_bbox[2], face_bbox[3],
                  frame, (original_gaze[1], original_gaze[0]), color=(0, 0, 255))  # Red color
    
    # Draw smoothed gaze with green color if available
    if face_bbox is not None and smoothed_gaze[0] is not None and smoothed_gaze[1] is not None:
        draw_gaze(face_bbox[0], face_bbox[1], face_bbox[2], face_bbox[3],
                  frame, (smoothed_gaze[1], smoothed_gaze[0]), color=(0, 255, 0))  # Green color
    
    # Add text information to the frame
    # AOI information

    aoi_name = get_aoi_name(aoi)
    cv2.putText(frame, f"AOI: {aoi_name}", (10, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                1, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Reading state
    cv2.putText(frame, f"State: {reading_state}", (10, 80), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                1, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Page number
    cv2.putText(frame, f"Page: {page_number}", (10, 110), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                1, (255, 255, 255), 1, cv2.LINE_AA)
    
    return frame
        
def get_transformations():
    """Create image transformations for gaze prediction"""
    return transforms.Compose([
        transforms.Resize(448),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


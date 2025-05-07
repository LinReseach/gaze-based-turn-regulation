
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 14:12:31 2025

@author: chenglinlin
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 11:23:39 2024

@author: chenglinlin
"""

import socket
import cv2
import numpy as np
from PIL import Image
import time




import cv2
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
# video_capture1 = cv2.VideoCapture(0)
video_capture2 = cv2.VideoCapture(1) # 0 for logitech, 1 for built in

video_capture2.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
width = 1920#3840
height = 1080#2160
video_capture2.set(cv2.CAP_PROP_FRAME_WIDTH, width)
video_capture2.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

a4k=[]
if __name__ == '__main__':
    
    i=0
    while True:
        #connect.adjust_head(-0.2, 0.0)
        # i=i+1
        # s=time.time() 
        img4k =  video_capture2.read()[1]
        a4k.append(img4k)
        cv2.imshow('pepper stream', img4k)
        print(i)
        i=i+1
        # c=s-time.time()
        # ci=c+ci
        # print(s-time.time())
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
        #cv2.imshow('pepper stream', img)
        cv2.waitKey(1)
    
    cv2.destroyAllWindows()
    
    


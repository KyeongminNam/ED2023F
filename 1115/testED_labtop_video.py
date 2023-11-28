import cv2
import os,sys
import numpy as np
import time
import tensorflow as tf
from tensorflow import keras


savedmodel = keras.models.load_model('./models/modelED_1122_v3.h5', compile=False)

font = cv2.FONT_HERSHEY_SIMPLEX 
org = (50, 50) 
fontScale = 1
color = (255, 255, 255) 
thickness = 2


def visualize_arrow(frame, a):
    if (a == 0): # straight
        cv2.arrowedLine(frame, (400, 150), (400, 100), color, thickness=2)
        
    elif(a == 1): # left2
        cv2.arrowedLine(frame, (170, 150), (120, 100), color, thickness=2)
        cv2.arrowedLine(frame, (200, 150), (150, 100), color, thickness=2)
        
    elif(a == 2): # left1
        cv2.arrowedLine(frame, (300, 150), (250, 100), color, thickness=2)
        
    elif(a == 3): # right2
        cv2.arrowedLine(frame, (500, 150), (550, 100), color, thickness=2)
        
    elif(a == 4): # left1
        cv2.arrowedLine(frame, (600, 150), (650, 100), color, thickness=2)
        cv2.arrowedLine(frame, (630, 150), (680, 100), color, thickness=2)
        





video_file = "testvideo.mp4" # 동영상 파일 경로

cap = cv2.VideoCapture(video_file) # 동영상 캡쳐 객체 생성

sig_stack = 0
prev_sig = 0
sig = 0 
stack = 0
send = 'straight'

if cap.isOpened(): 
    while(True):

        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #cv2.imshow("Camera",gray)

        gray_resize = cv2.resize(gray, (48,48))
        normalized = np.zeros((48,48))
        normalized = cv2.normalize(gray_resize, normalized, 0, 255, cv2.NORM_MINMAX)
            
        pixel_values = normalized
        pixel_values = 1.0 * np.asarray(pixel_values)
        pixel_values = pixel_values / 255
        X_in = np.reshape(pixel_values, [1, 48, 48, 1])


        predict = savedmodel.predict(X_in) 
        Y = np.argmax(predict,axis=1)
        cv2.putText(gray, str(Y), org, font, fontScale, color, thickness, cv2.LINE_AA) 


        # Vibration signal -----------------------------------------
        label = ['straight', 'left2', 'left1', 'right2', 'right1']
        sig = int(Y)
        stack += 1
        
        if (prev_sig == sig):
            sig_stack += 1
            
            if (sig_stack >=30):
                send = label[sig]
                visualize_arrow(gray, sig)
                sig_stack = 0
                stack = 0
        else:
            sig_stack = 0
        
        if (stack >= 30):
            visualize_arrow(gray, sig)
            stack = 0
        
        prev_sig = sig   
        # Vibration signal -----------------------------------------
        
        cv2.imshow("Camera",gray)
        cv2.waitKey(10)
        
else:
    print("can't open video.")
    
    
cap.release()
cv2.destroyAllWindows()
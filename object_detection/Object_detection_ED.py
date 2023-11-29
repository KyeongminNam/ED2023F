### Picamera Object Detection Using Tensorflow Classifier ###
#
# Author: Sahil Kumar
# Date: 06/09/2020
# Description: 
# This program uses a TensorFlow classifier to perform object detection.
# It loads the classifier uses it to perform object detection on a Picamera feed.
# It draws boxes and scores around the objects of interest in each frame from
# the Picamera. It also can be used with a webcam by adding "--usbcam"
# when executing this script from the terminal.

## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb


# Import packages
import os
import cv2
import numpy as np
#from picamera.array import PiRGBArray
#from picamera import PiCamera
from picamera2 import Picamera2, Preview
from datetime import datetime

import time
import tensorflow as tf
import argparse
import sys

# Set up camera constants
#IM_WIDTH = 1280
#IM_HEIGHT = 720
IM_WIDTH = 640    #Use smaller resolution for
IM_HEIGHT = 480   #slightly faster framerate

# Select camera type (if user enters --usbcam when calling this script,
# a USB webcam will be used)
camera_type = 'picamera'
parser = argparse.ArgumentParser()
parser.add_argument('--usbcam', help='Use a USB webcam instead of picamera',
                    action='store_true')
args = parser.parse_args()
if args.usbcam:
    camera_type = 'usb'

# This is needed since the working directory is the object_detection folder.
sys.path.append('..')

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'data','mscoco_label_map.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 90

## Load the label map.
# Label maps map indices to category names, so that when the convolution
# network predicts `5`, we know that this corresponds to `airplane`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

possible_danger_list = [1, 2, 3, 4, 15, 16, 17, 18, 37, 62]

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.compat.v1.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.compat.v1.Session(graph=detection_graph)


# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()
font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize camera and perform object detection.
# Set GPIO and Vibration motor

import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BCM)

leftpin = 5
straightpin= 6
rightpin = 13
GPIO.setmode(GPIO.BCM)
GPIO.setup(leftpin, GPIO.OUT)
GPIO.setup(straightpin, GPIO.OUT)
GPIO.setup(rightpin, GPIO.OUT)

def vib1(pinnum):
    GPIO.output(pinnum, GPIO.HIGH)
    time.sleep(0.1)
    GPIO.output(pinnum, GPIO.LOW)
    time.sleep(0.5)

def vib2(pinnum):
    GPIO.output(pinnum, GPIO.HIGH)
    time.sleep(0.1)
    GPIO.output(pinnum, GPIO.LOW)
    time.sleep(0.1)
    GPIO.output(pinnum, GPIO.HIGH)
    time.sleep(0.1)
    GPIO.output(pinnum, GPIO.LOW)
    time.sleep(0.5)

def vib3(pinnum):
    GPIO.output(pinnum, GPIO.HIGH)
    time.sleep(0.1)
    GPIO.output(pinnum, GPIO.LOW)
    time.sleep(0.1)
    GPIO.output(pinnum, GPIO.HIGH)
    time.sleep(0.1)
    GPIO.output(pinnum, GPIO.LOW)
    time.sleep(0.1)
    GPIO.output(pinnum, GPIO.HIGH)
    time.sleep(0.1)
    GPIO.output(pinnum, GPIO.LOW)
    time.sleep(0.5)




### Picamera ###
camera = Picamera2()
camera.start()


## clssify direction
th1 = 213 # 640/3
th2 = 427

senddir = 0 # direction 0=left 1=straight 2=right
dir_label = ['left', 'straight', 'right']

hth1 = 150
hth2 = 330
sendheight = 0 # height 0=small 1=mid 2=large
height_label = ['small', 'mid', 'large']

video_name = "./videos/" + str(datetime.now()) + ".mp4"

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(video_name, fourcc, 3.0, (640,480))

while True:
    t1 = cv2.getTickCount()

    frame = camera.capture_array()
    frame1 = np.copy(frame)
    frame1.setflags(write =1)
    frame_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    frame_expanded = np.expand_dims(frame_rgb, axis = 0)

    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')                                                                                                       
    num = detection_graph.get_tensor_by_name('num_detections:0')

    

    (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections], feed_dict={image_tensor: frame_expanded})

    if np.squeeze(classes).astype(np.int32)[0] in possible_danger_list:
    
        # Draw the results of the detection (aka 'visulaize the results')
        vis_util.visualize_boxes_and_labels_on_image_array(
            frame_rgb,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8
            #min_score_thresh=0.40
            )

        coordinates = vis_util.return_coordinates(
                            frame_rgb,
                            np.squeeze(boxes),
                            np.squeeze(classes).astype(np.int32),
                            np.squeeze(scores),
                            category_index,
                            use_normalized_coordinates=True,
                            line_thickness=8
                            )
        # print(category_index[np.squeeze(classes).astype(np.int32)[0]]["name"])

        # obstacle detecting
        if (len(coordinates) != 0):
            obstacle_position = [0,0,0]
            for coord_arr in coordinates:
                objcenter = (int(coord_arr[0]), int(coord_arr[1]))
                objheight = int(coord_arr[2])

                cv2.circle(frame_rgb, objcenter, 10, (0,0,255), -1) # visualize obstacle center

                # object location detecting ---------------------------
                if(objcenter[0] < th1):
                    senddir = 0
                    obstacle_position[0] = 1
                elif((th1 < objcenter[0]) and (objcenter[0] < th2)):
                    senddir = 1
                    obstacle_position[1] = 1

                elif(th2 < objcenter[0]):
                    senddir = 2
                    obstacle_position[2] = 1
                else:
                    print('Out Of Range')

                cv2.putText(frame_rgb,"direction: " + dir_label[senddir] ,(int(coord_arr[0])-50, int(coord_arr[1])+50),font,0.7,(0,0,255),2,cv2.LINE_AA)

                # object size detecting ---------------------------
                if(objheight < hth1):
                    sendheight = 0
                elif((hth1 < objheight) and (objheight < hth2)):
                    sendheight = 1
                elif(hth2 < objheight):
                    sendheight = 2
                else:
                    print('Out Of Range')

                cv2.putText(frame_rgb,"size: " + height_label[sendheight],(int(coord_arr[0])-50, int(coord_arr[1])+80),font,0.7,(0,0,255),2,cv2.LINE_AA)

                # vibration signal ---------------------------
                if(senddir==0): # left
                    if(sendheight==0): # small
                        vib1(leftpin)
                    elif(sendheight==1): # mid
                        vib2(leftpin)
                    elif(sendheight==2): # large
                        vib3(leftpin)
                    else:
                        print('Out Of Range')
                elif(senddir==1): # striaght
                    if(sendheight==0): # small
                        vib1(straightpin)
                    elif(sendheight==1): # mid
                        vib2(straightpin)
                    elif(sendheight==2): # large
                        vib3(straightpin)
                    else:
                        print('Out Of Range')
                elif(senddir==2): # right
                    if(sendheight==0): # small
                        vib1(rightpin)
                    elif(sendheight==1): # mid
                        vib2(rightpin)
                    elif(sendheight==2): # large
                        vib3(rightpin)
                    else:
                        print('Out Of Range')


            # for i in range(len(obstacle_position)):
            #     if obstacle_position[i] == 1:
            #         gpio.output(pin_list[i], True)
            #     else:
            #         gpio.output(pin_list[i], False)

                # vibration signal ---------------------------
    else:
        pass




    # show fps
    cv2.putText(frame_rgb,"FPS: {0:.2f}".format(frame_rate_calc),(30,50),font,1,(255,255,0),1,cv2.LINE_AA) 

    # All the results have been drawn on the frame, so it's time to display it.
    cv2.line(frame_rgb, (th1,0), (th1,480), (0,0,255), 2) # dividing line
    cv2.line(frame_rgb, (th2,0), (th2,480), (0,0,255), 2) # dividing line
    cv2.imshow('Object detector', frame_rgb)
    out.write(frame_rgb)

    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc = 1/time1
    

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break



camera.close()
      
out.release()
cv2.destroyAllWindows()

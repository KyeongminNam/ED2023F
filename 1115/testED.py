import cv2
import os,sys
import numpy as np
import time
#from picamera2 import Picamera2, Preview
from picamera2 import *
import tensorflow as tf
from tensorflow import keras

picam2 = Picamera2()
camera_config = picam2.create_still_configuration(main={"size": (1920, 1080)}, lores={"size": (640, 480)}, display="lores")
picam2.configure(camera_config)
picam2.start()

savedmodel = keras.models.load_model('./modelED_v1.h5', compile=False)

font = cv2.FONT_HERSHEY_SIMPLEX 
org = (50, 50) 
fontScale = 1
color = (255, 0, 0) 
thickness = 2



while(True):

    im = picam2.capture_array()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
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
    print(Y)
    cv2.putText(gray, str(Y), org, font, fontScale, color, thickness, cv2.LINE_AA) 
    cv2.imshow("Camera",gray)

    cv2.waitKey(100)


    
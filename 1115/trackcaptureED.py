import cv2
import os,sys
import numpy as np
import time
#from picamera2 import Picamera2, Preview
from picamera2 import *


picam2 = Picamera2()
camera_config = picam2.create_still_configuration(main={"size": (1920, 1080)}, lores={"size": (640, 480)}, display="lores")
picam2.configure(camera_config)
picam2.start()


path = os.getcwd()
os.system("mkdir dataED")
label = ['straight', 'left2', 'left1', 'right2', 'right1']

for i in label:
    counter = 0
    try:
        os.system("mkdir %s/dataED/%s" %(path,i))
    except:
        pass
    
    fpath = os.path.join(os.getcwd(),'dataED',i)
    input("Current Label : %s \n Press enter to contine..." %i)

    while (counter!=100):

        #ret, frame = cap.read()
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #cv2.imshow("Camera",gray)

        im = picam2.capture_array()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Camera",gray)

        gray_resize = cv2.resize(gray, (48,48))
        normalized = np.zeros((48,48))
        normalized = cv2.normalize(gray_resize, normalized, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite(r"./dataED/%s/%d.jpg" %(i,counter), normalized)
        print("%s\\%d.jpg" %(fpath,counter))
        print('Saved %s, %d' %(i,counter))
        counter += 1
        cv2.waitKey(100)



cv2.destroyAllWindows()
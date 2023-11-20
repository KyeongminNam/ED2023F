
import numpy as np
import os
from PIL import Image

filedirs = os.listdir(os.path.join(os.getcwd(),'dataED'))
expension = '*.jpg'
txtdir = os.path.join(os.getcwd(),'dataED', 'txt')
os.system('mkdir %s' %txtdir)
label_file = open('LabelED' + '.csv', 'w')
image_counter = 0

given_class = ['hard coded to match 2017version', 'straight', 'left2', 'left1', 'right2', 'right1']

for d in filedirs:
    filedir = os.path.join(os.path.join(os.getcwd(),'dataED', d))
    filenames = [f for f in os.listdir(filedir) if f.endswith(".jpg")]
    for name in filenames: #This for loop sets file names and writes pixel values in a txt file, and lists them in Label.csv.

        if image_counter / 10000 >= 1:
            image_text = str(image_counter)
        elif image_counter / 1000 >= 1:
            image_text = '0' + str(image_counter)
        elif image_counter / 100 >= 1:
            image_text = '00' + str(image_counter)
        elif image_counter / 10 >= 1:
            image_text = '000' + str(image_counter)
        else:
            image_text = '0000' + str(image_counter)

        im = Image.open(os.path.join(filedir,name))

        pixel_values = list(im.getdata())
        pixel_values = 1.0 * np.asarray(pixel_values)
        pixel_values = pixel_values / 255
        pixel_values = pixel_values.tolist()
        pixel_values = np.str_(pixel_values)
        
        real_values = given_class.index(d)
        
        filename = open(os.path.join(txtdir,image_text) + '.txt', 'w')
        filename.write(pixel_values)
        filename.close()
        label_file.write(image_text + '.txt' + ',%d\n' %real_values)
        image_counter += 1

label_file.close()


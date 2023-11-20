from glob import glob
import os
import numpy as np
import sys
import csv
import gc

class DataSet:

    def __init__(self, filedir, expension, label_filedir, label_file, data_size):
        ## parameters initialization
        self.dataset = [None] * data_size
        self.label_dataset = []
        self.ShuffledIndex = []
        self.filedir = filedir
        self.expension = expension
        self.Shuffle = False
        self.label_filedir = label_filedir
        self.label_file = label_file
        self.counter = 0
        if self.expension == '.txt':
            self.expension = '*.txt'
        elif self.expension == '.jpg':
            self.expension == '*.jpg'
        else:
            print('extension error')
            sys.exit(1)


        filenames = glob(os.path.join(self.filedir,self.expension))
        label_filename = glob(os.path.join(self.label_filedir,self.label_file))

        filenames.sort()

        for fname in filenames: #Reads txt file names, and saves pixel values in an array of size 2304 (=48*48).
            a = open(fname, 'r')
            b = a.read().split(',')
            b[0] = b[0][1:]
            b[2303] = b[2303][0:-1]
            self.dataset[self.counter] = b
            a.close()
            self.counter += 1
            if self.counter % 1000 == 0:
                gc.collect()

        temp = open(label_filename[0], 'r')
        self.label_file = csv.reader(temp)

        for row in self.label_file: #Opens Label.csv and assigns a vector according to its label.
            ## if you want to adjust this file to something else, modify this section
            if row[1] == '1': #straight
                one_hot = [1, 0, 0, 0, 0]
            elif row[1] == '2': #left2
                one_hot = [0, 1, 0, 0, 0]
            elif row[1] == '3': #left1
                one_hot = [0, 0, 1, 0, 0]
            elif row[1] == '4': #right2
                one_hot = [0, 0, 0, 1, 0]
            elif row[1] == '5': #right1
                one_hot = [0, 0, 0, 0, 1]


            self.label_dataset.append(one_hot)

        temp.close()

        for i in range(0, len(self.dataset)):
            self.dataset[i] = list(map(float, self.dataset[i]))


    def next_batch(self, BatchSize, shuffle = False):#Randomly picks sample images. One batch is a size of "BatchSize". If BatchSize=50, randomly picks 50 images from the dataset.

        if shuffle == True:

            self.index = np.arange(0, len(self.dataset))

            if shuffle == True: # shuffle
                #np.random.shuffle(self.index)
                self.ShuffledIndex = self.index
            else:
                self.ShuffledIndex = self.index

        self.NextIndex = self.ShuffledIndex[:BatchSize]
        self.ShuffledIndex = self.ShuffledIndex[BatchSize:]

        self.NextBatchData = [self.dataset[i] for i in self.NextIndex]
        self.NextBatchLabels = [self.label_dataset[i] for i in self.NextIndex]


        return self.NextBatchData, self.NextBatchLabels

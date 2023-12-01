import tensorflow as tf
import dataset_makerED as DM
import numpy as np
import os, sys
from tensorflow import keras

filedir = os.path.join(os.getcwd(), 'dataED','txt','')

expension = '.txt'
print(filedir)
#print(os.listdir(filedir))
label_filedir = os.getcwd()
label_file = 'LabelED.csv'
num_data=len(os.listdir(filedir))  #This number should equal the number of sampled images.

data = DM.DataSet(filedir=filedir, expension=expension, label_filedir=label_filedir, label_file=label_file, data_size=num_data)   #data size = the number of images in data folder


model = keras.models.Sequential([
    keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', input_shape=(48, 48, 1)),
    keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same'),
    
    keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'),
    keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same'),
    
    keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'),
    keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same'),
    
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(5, activation='softmax')
])
#https://www.kaggle.com/code/amyjang/tensorflow-mnist-cnn-tutorial


model.compile(
    optimizer=keras.optimizers.Adam(0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

train_images=np.zeros(48*48).reshape(1, 48,48,1)
train_labels=np.zeros(5).reshape(1,5)


batch_size = 1
total_batch = int(num_data / batch_size)

'''
for epoch in range(20):
    total_cost = 0
    data_shuffle = True
    accumulate_accuracy = 0
    for i in range(total_batch):
        batch_xs, batch_ys = data.next_batch(batch_size, data_shuffle)
        batch_xs = np.asarray(batch_xs)
        batch_ys = np.asarray(batch_ys)
        batch_xs = batch_xs.reshape(-1, 48, 48, 1)

        train_images = np.vstack((train_images, batch_xs))
        train_labels = np.vstack((train_labels, batch_ys))
        
        
        data_shuffle = False

    data_shuffle_test = True
'''
data_shuffle = True
batch_size = num_data
batch_xs, batch_ys = data.next_batch(batch_size, data_shuffle)
batch_xs = np.asarray(batch_xs)
batch_ys = np.asarray(batch_ys)
batch_xs = batch_xs.reshape(-1, 48, 48, 1)



train_images=np.delete(train_images, 0, axis=0)
train_labels=np.delete(train_labels, 0, axis=0)


print(batch_xs.shape)
print(batch_ys.shape)
print('Data collected!')

model.fit(batch_xs, batch_ys, epochs=10)

model.save("./modelED_1115_v5.h5")
print('Training finish!')

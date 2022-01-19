# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 08:54:29 2022

@author: Student
"""

import directories_files
import numpy as np
import pandas as pd
import os
import itertools
import os, stat, time
from os.path import dirname as up
import shutil
from PIL import Image
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# import cv2

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import SGD, RMSprop, Adam, Adagrad, Adadelta
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Conv2D, MaxPool2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping
import keras_preprocessing.image as IMAGE

import random

random.seed(0)

class NN:
	def __init__(self):
		self.model = Sequential()
		self.model.add(Conv2D(filters=32,kernel_size=(4,4),input_shape=(50,50,3),activation='relu'))
		self.model.add(MaxPool2D(pool_size=(2,2)))
		self.model.add(MaxPool2D(pool_size=(2,2)))
		self.model.add(Dropout(0.25))
		self.model.add(Conv2D(filters=32,kernel_size=(4,4),activation='relu'))
		self.model.add(MaxPool2D(pool_size=(2,2)))
		self.model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
		self.model.add(Dropout(0.25))
		self.model.add(Flatten())
		self.model.add(Dense(64,activation='relu'))
		self.model.add(Dense(1,activation='sigmoid'))
		self.model.compile(loss = 'binary_crossentropy', optimizer ='adam', metrics= ['accuracy'])		
		self.model.summary()

base = r'D:/breast cancer dataset/'

ids = os.listdir(base)
data0 = []
data1 = []
for id in tqdm(ids):
  try:
    files1 = os.listdir(base + id + '/1/')
    files0 = os.listdir(base + id + '/0/')
    for x in files1:
      data1.append(base + id + '/1/' + x)
    for x in files0:
      data0.append(base + id + '/0/' + x)
  except:
    FileNotFoundError
print(len(data1))
print(len(data0))

random.shuffle(data1)
data1 = data1[:20000]
len(data1)

images=[]
labels=[]

for i in tqdm(data1):
  label = int(i[-5])
  img = IMAGE.img_to_array(IMAGE.load_img(i, target_size=(50, 50)))
  images.append(img)
  labels.append(label)

y = np.array(labels)
x = np.stack(images)/255

print(np.unique(y, return_counts=True))

x_train,x_test,y_train,y_test = train_test_split(x, y, random_state=0, test_size=0.3)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


network = NN()


def main():
	print("hello")

if __name__== '_main_':
	main()



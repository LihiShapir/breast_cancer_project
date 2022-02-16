# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 08:54:29 2022

@author: Student
"""

#import BuildDataset.py as bd
# import directories_files
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
import cv2

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import SGD, RMSprop, Adam, Adagrad, Adadelta
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Conv2D, MaxPool2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping
import keras_preprocessing.image as IMAGE

import random

from glob import glob

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
		
def load_data(path):
	path = path + r'\**\*.png'
	imagePatches = glob(path , recursive=True)
	images = []
	labels = []
	
	for i in range(len(imagePatches)):
		image_path = imagePatches[i]
		file=image_path.split(os.path.sep)[-1]
		label=int(file[-5:-4])
		
		img = cv2.imread(image_path)
		images.append(img)
		labels.append(label)
		
		if i % 1000 == 0:
			print(i)
	return images, labels
		
	


#base = r'D:/breast cancer dataset/'
def Build_model():
	imagePatches = glob(r"C:\Users\Student\OneDrive\Downloads\archive (4)\**\*.png" , recursive=True)
	print("imgs:"+ str(len(imagePatches)))
	for filename in imagePatches[0:10]:
		print(filename)
	data0 = []
	data1 = []
	for filename in imagePatches:
		if filename.endswith("class0.png"):
			data0.append(filename)
		else:
			data1.append(filename)
	print("data0:"+ str(len(data0)))
	print("data1:"+ str(len(data1)))
	sampled_data0 = random.sample(data0, 78786)		
	sampled_data1 = random.sample(data1, 78786)
	print("data0:"+ str(len(sampled_data0)))
		
"""

#network = NN()
"""

def main():
	# Build_model()
	x_train, y_train = load_data(r'D:\data\training')
	network = NN()
	network.model.fit(x_train, y_train)
	

main()

#if __name__== '_main_':
#	main()



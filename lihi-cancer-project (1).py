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
		"""
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
		"""
		
		self.model = Sequential()
		self.model.add(Conv2D(filters=32,kernel_size=(3,3),input_shape=(50,50,3),activation='relu')) # Changed kernel size to 3x3
		self.model.add(MaxPool2D(pool_size=(2,2)))
		self.model.add(Dropout(0.25))
		self.model.add(Conv2D(filters=32,kernel_size=(3,3),activation='relu')) # Changed kernel size to 3x3
		self.model.add(MaxPool2D(pool_size=(2,2)))
		self.model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
		self.model.add(Dropout(0.25))
		self.model.add(Flatten())
		self.model.add(Dense(64,activation='relu'))
		self.model.add(Dense(1,activation='sigmoid'))
		self.model.compile(loss = 'binary_crossentropy', optimizer ='adam', metrics= ['accuracy'])		
		self.model.summary()
	def train(self, x, y):
		return self.model.fit(x, y, batch_size=32, epochs=10)
'''
def load_data(path):
	"""
		Load images from every subdirectory of path
		
		Returns shuffled loaded images arrays and labels
	"""
	path = path + r'\\**\\*.png'
	imagePatches = glob(path , recursive=True)#?
	images = []
	labels = []
	
	#for i in range(len(imagePatches)):
	for i in range(2):
		image_path = imagePatches[i]
		print('image_path is = ' , image_path.shape)
		file=image_path.split(os.path.sep)[-1]
		
		# Load the image and label
		try:
			#img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
			img = cv2.cvtColor(cv2.imread(image_path))
			#img = np.expand_dims(img, axis=0)
			img = img / 255
			
		except:
			print(image_path)
			continue
			
		label=int(file[-5:-4])
		
		images.append(img)
		labels.append(label)
		if i % 1000 == 0:
			print(i)

	x, y = np.array(images), np.array(labels)
	
	# Shuffle the data
	idx = np.random.permutation(len(images)) 
	x, y = [x[i] for i in idx], [y[i] for i in idx]
	return x, y
'''
#base = r'D:/breast cancer dataset/'
#not in use yet
def Build_model():
	imagePatches = glob(r"C:\Users\Student\OneDrive\Downloads\archive (4)\**\*.png" , recursive=True)
	print("imgs:"+ str(len(imagePatches)))
	#for filename in imagePatches[0:10]:
	#	print(filename)
	data0 = []
	data1 = []
	for filename in imagePatches:
		if filename.endswith("class0.png"):
			data0.append(filename)
		else:
			data1.append(filename)
			
	print("Total Data:")
	print("data0: "+ str(len(data0)))
	print("data1: "+ str(len(data1)))
	print()
	
	sampled_data0 = random.sample(data0, 78786)		
	sampled_data1 = random.sample(data1, 78786)
	BASE_PATH = r"D:\DATAFORLIHI"
	TRAIN_PATH = os.path.sep.join([BASE_PATH, "training"])
	VAL_PATH = os.path.sep.join([BASE_PATH, "validation"])
	TEST_PATH = os.path.sep.join([BASE_PATH, "testing"])

	TRAIN_SPLIT = 0.8
	VAL_SPLIT = 0.1
	
	train_count = int(len(sampled_data0)*TRAIN_SPLIT)
	val_count = int(len(sampled_data0)*VAL_SPLIT)
	
	trainPaths0 = sampled_data0[:train_count]
	valPaths0 = sampled_data0[train_count:train_count + val_count]
	testPaths0 = sampled_data0[train_count + val_count:]
	
	trainPaths1 = sampled_data1[:train_count]
	valPaths1 = sampled_data1[train_count:train_count + val_count]
	testPaths1 = sampled_data1[train_count + val_count:]
	
	trainPaths = trainPaths0 + trainPaths1
	valPaths = valPaths0 + valPaths1
	testPaths = testPaths0 + testPaths1
	
	print("Train paths: " + str(len(trainPaths)))
	print("Test paths: " + str(len(testPaths)))
	print("Val paths: " + str(len(valPaths)))
	
	datasets=[("training", trainPaths, TRAIN_PATH),
		      ("validation", valPaths, VAL_PATH),
	          ("testing", testPaths, TEST_PATH)
			  ]
	
	for (setType, originalPaths, basePath) in datasets:
		print(f'Building {setType} set')

		if not os.path.exists(basePath):
			print(f'Building directory {basePath}')
			os.makedirs(basePath)

		for path in originalPaths:
			file=path.split(os.path.sep)[-1]
			label=file[-5:-4]
			
			labelPath=os.path.sep.join([basePath,label])
			if not os.path.exists(labelPath):
				print(f'Building directory {labelPath}')
				os.makedirs(labelPath)

			newPath=os.path.sep.join([labelPath, file])
			shutil.copy2(path, newPath)
	
	
	
	#index=int(len(sampled_data0)*TRAIN_SPLIT)
	#trainPaths=originalPaths[:index]
	#testPaths=originalPaths[index:]

#	index=int(len(trainPaths)*VAL_SPLIT)
	#valPaths=trainPaths[:index]
	#trainPaths=trainPaths[index:]
	

#network = NN()


def main():
	Build_model()
	#x_train, y_train = load_data(r'D:\data\training')
	
"""   
	network = NN()
	# network.train(x_train[0], y_train[0])
	print(network.model.predict(x_train[0]))
"""	

# main()

if __name__== '__main__':
	main()



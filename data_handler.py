# -*- coding: utf-8 -*-
"""
Lihi Shapir- Breast cancer project

"""

from glob import glob
import os
import random
import cv2
import shutil
from tkinter import filedialog 
import zipfile
import numpy as np
from tensorflow.keras.utils import to_categorical


class Datahandler:
    def __init__(self):
        self.extracted_dir = None  # The directory of the extracted data
        self.data0 = None  # The paths to the images with label 0
        self.data1 = None  # The paths to the images with label 1
        self.train_paths = None  # The paths to the train images
        self.test_paths = None  # The paths to the test images
        self.val_paths = None  # The paths to the validation images
        self.split_dir = None  # The path to the directory of the split data
    
    def extract_data(self) -> None :
        """
        Extract the images from the dataset zip to an empty folder
        The user chooses the dataset zip
        The user chooses the empty directory
        """
        print('Choose the archive_final zip')
        zip_file = filedialog.askopenfilename(initialdir=os.getcwd(), filetypes=(('ZIP File', '*.zip'),), title='Choose the archive_final zip')
        self.extracted_dir = Datahandler.__ask_empty_directory('Choose an empty directory for the extracted data', title='Choose an empty directory for the extracted data')
        
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(self.extracted_dir)
        self.delete_corrupted()
    
    def delete_corrupted(self) -> None :
        """
        Deletes the corrupted images from the extracted directory and
        appends the images paths to the right data list
        """
        print('Deleting corrupted data')
        if self.extracted_dir is None:
            self.extracted_dir = Datahandler.__ask_directory('Choose the extracted data directory', title='Choose the extracted data directory')
        imagePatches = glob(self.extracted_dir + r"\**\*.png" , recursive=True)
        print("imgs:"+ str(len(imagePatches)))
        self.data0 = []
        self.data1 = []
        
        corrupted_0 = 0
        corrupted_1 = 0
        for filename in imagePatches:
            is0 = filename.endswith("class0.png")
            if Datahandler.__is_corrupted(filename):
                if is0:
                    corrupted_0 += 1
                else:
                    corrupted_1 += 1
                if os.path.exists(filename):
                    os.remove(filename) # one file at a time
            else:
                if is0:
                    self.data0.append(filename)
                else:
                    self.data1.append(filename)
            
            
        print("Corrupted Data:")
        print("data0: "+ str(corrupted_0))
        print("data1: "+ str(corrupted_1))
        print()
        
        print("Total Data:")
        print("data0: "+ str(len(self.data0)))
        print("data1: "+ str(len(self.data1)))
        print()
        self.split_data()
    
    def split_data(self, train_size=0.7, test_size=0.2) -> None :
        """
        Split the data to train, test and validation data according to the
        train_size and test-size arguments
        The user chooses The empty folder to split the data in
        """
        print(f'Splitting data to {train_size} train, {test_size} test and {1 - train_size - test_size}')
        self.split_dir = Datahandler.__ask_empty_directory('Choose an empty directory for the split data ')
       
        sampled_data0 = random.sample(self.data0, min(len(self.data0), len(self.data1)))        
        sampled_data1 = random.sample(self.data1, min(len(self.data0), len(self.data1)))
        
        print("Sampled Data:")
        print("data0: "+ str(len(sampled_data0)))
        print("data1: "+ str(len(sampled_data1)))
        print()
        
        TRAIN_PATH = os.path.sep.join([self.split_dir, "training"])
        VAL_PATH = os.path.sep.join([self.split_dir, "validation"])
        TEST_PATH = os.path.sep.join([self.split_dir, "testing"])
        
        train_count = int(len(sampled_data0)*train_size)
        val_count = int(len(sampled_data0)*(1 - train_size - test_size))
        
        trainPaths0 = sampled_data0[:train_count]
        valPaths0 = sampled_data0[train_count:train_count + val_count]
        testPaths0 = sampled_data0[train_count + val_count:]
        
        trainPaths1 = sampled_data1[:train_count]
        valPaths1 = sampled_data1[train_count:train_count + val_count]
        testPaths1 = sampled_data1[train_count + val_count:]
        
        self.train_paths = trainPaths0 + trainPaths1
        self.val_paths = valPaths0 + valPaths1
        self.test_paths = testPaths0 + testPaths1
        
        print("Train paths: " + str(len(self.train_paths)))
        print("Test paths: " + str(len(self.test_paths)))
        print("Val paths: " + str(len(self.val_paths)))
        
        datasets=[("training", self.train_paths, TRAIN_PATH),
                  ("validation", self.val_paths, VAL_PATH),
                  ("testing", self.test_paths, TEST_PATH)
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
        print("finished splitting the data")    
    
    def choose_split_dir(self) -> None :
        self.split_dir = Datahandler.__ask_directory('Choose the split data directory', title='Choose the split data directory')
    
    def load_train(self):
        """
        Loads the images and labels for the training session
        return: 2 arrays that contains the images and labels for train
        """
        print('Loading train data')
        if self.train_paths is None:
            if self.split_dir is None:
                self.split_dir = Datahandler.__ask_directory('Choose the split data directory', title='Choose the split data directory')
            TRAIN_PATH = os.path.sep.join([self.split_dir, "training"])
            self.train_paths = glob(TRAIN_PATH + r"\**\*.png" , recursive=True)
        
        images_list = []
        labels_list = []
        count = 1
        for image_path in self.train_paths:
            img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            img = img / 255 
            label=int(image_path[-5:-4])
            
            images_list.append(img)
            labels_list.append(to_categorical(label, num_classes=2))
            
            if count % 1000 == 0:
                print(count)
            count += 1
            
        idx = np.random.permutation(len(images_list))
        x, y = [images_list[i] for i in idx], [labels_list[i] for i in idx]
        x, y = np.array(x), np.array(y)
        return x, y
    
    def load_test(self):
        """
        Loads the images and labels for the testing session
        return: 2 arrays that contains the images and labels for test
        """
        print('Loading test data')
        if self.test_paths is None:
            if self.split_dir is None:
                self.split_dir = Datahandler.__ask_directory('Choose the split data directory', title='Choose the split data directory')
            PATH = os.path.sep.join([self.split_dir, "testing"])
            self.test_paths = glob(PATH + r"\**\*.png" , recursive=True)
        
        images_list = []
        labels_list = []
        count = 1
        for image_path in self.test_paths:
            img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            img = img / 255
            label=int(image_path[-5:-4])
            
            images_list.append(img)
            labels_list.append(to_categorical(label, num_classes=2))
            
            if count % 1000 == 0:
                print(count)
            count += 1
            
        idx = np.random.permutation(len(images_list))
        x, y = [images_list[i] for i in idx], [labels_list[i] for i in idx]
        x, y = np.array(x), np.array(y)
        return x, y
    
    def load_val(self):
        """
        Loads the images and labels for the evaluation
        return: 2 arrays that contains the images and labels for validation
        """
        print('Loading validation data')
        if self.val_paths is None:
            if self.split_dir is None:
                self.split_dir = Datahandler.__ask_directory('Choose the split data directory', title='Choose the split data directory')
            PATH = os.path.sep.join([self.split_dir, "validation"])
            self.val_paths = glob(PATH + r"\**\*.png" , recursive=True)
        
        images_list = []
        labels_list = []
        count = 1
        for image_path in self.val_paths:
            img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            img = img / 255
            label=int(image_path[-5:-4])
            
            images_list.append(img)
            labels_list.append(to_categorical(label, num_classes=2))
            
            if count % 1000 == 0:
                print(count)
            count += 1
            
        idx = np.random.permutation(len(images_list))
        x, y = [images_list[i] for i in idx], [labels_list[i] for i in idx]
        x, y = np.array(x), np.array(y)
        return x, y
    
    @staticmethod
    def __is_corrupted(img_path):
        """
        Check if the image in the given path is corrupted
        Checks if there is an error while loading and normalizing the image
        Checks if the image shape is 50x50
        
        Returns True if the image is corrupted, False otherwise
        """
        try:
            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            img = img / 255
                
        except Exception as e:
            #in case of an non usable image return True
            print(e)
            print(img_path)
            return True
            
        if not img.shape == (50, 50, 3):
            print(img.shape)
            print(img_path)
            return True
        
        return False
           
    @staticmethod
    def __ask_directory(message='', **kwargs):
        """
        Ask the user to choose a directory
        message: The message to send when asking for a directory
        return: the directory that the use choose
        """
        if not message == '':
            print(message)
        dire = filedialog.askdirectory(initialdir=os.getcwd(), **kwargs)
        while dire == '':
            dire = filedialog.askdirectory(initialdir=os.getcwd(), **kwargs)
        return dire
    
    @staticmethod
    def ask_empty_directory(message, **kwargs):
        """
        Ask the user to choose an empty directory
        message: The message to send when asking for a directory
        return: the empty directory that the use choose
        """
        dire = Datahandler.__ask_directory(message, **kwargs)
        while not len(os.listdir(dire)) == 0:
            dire = Datahandler.__ask_directory(message, **kwargs)
        return dire
    
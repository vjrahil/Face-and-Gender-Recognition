#importing libraries
import numpy as np
import cv2
import os
import math
import dlib
import imutils
import face_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

#return list of images from a folder
def input(n):
        name = n
        images = []
        for filename in os.listdir(os.getcwd() + "/" + name):
            image = cv2.imread(os.getcwd() + "/" + name + "/" + filename, 1)
            images.append(image)
        return images

#train_test splitting for PCA and LDA
def split():
	X_train = []
	X_test = []
	Y_train = []
	Y_test = []
	directory = "output/"
	name = 0
	le = LabelEncoder()
	data = le.fit_transform(os.listdir(directory))
	for x in data:
		for i in range(16):
			Y_train.append(x)
		for i in range(4):
			Y_test.append(x)

	for x in os.listdir(directory):
			r = input(directory+x)
			name = 0
			for img in r:
				if name < 16:
					X_train.append(img)
				else:
					X_test.append(img)
				name += 1

	return X_train, X_test, Y_train, Y_test

#train_test splitting for gender recognition
def split2():
	X_train = []
	X_test = []
	Y_train = []
	Y_test = []
	arr_1 = []
	arr_2 = []
	directory = "output1/male/"
	data = os.listdir(directory)
	for x in data:
		r = input(directory+x)
		for img in r:
			arr_1.append(img)
			arr_2.append(1)

	directory = "output1/female/"
	data = os.listdir(directory)
	for x in data:
		r = input(directory+x)
		for img in r:
			arr_1.append(img)
			arr_2.append(2)

	return train_test_split(arr_1, arr_2, test_size = 0.2)

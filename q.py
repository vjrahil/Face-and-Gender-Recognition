#importing libraries
import numpy as np
import cv2
import os
import math
import dlib
import imutils
import face_utils

#preprocessing the images
def preprocess(img,name):
	image = img
	image = imutils.resize(image, width=500)
	shape = face_utils.points_68(image,0)
	#loop over the face detections
	cen_1 = ( (shape[36][0]+ shape[39][0])/2, (shape[36][1] + shape[39][1])/2)
	cen_2 = ( (shape[42][0]+ shape[45][0])/2, (shape[42][1] + shape[45][1])/2)
	
	#setting distance between center of eyes = 128 pixels
	distance = math.sqrt(pow(cen_1[0]-cen_2[0],2) + pow(cen_1[1] - cen_2[1],2))
	if distance == 0 or distance > 128 or shape[29][0] < 128 or shape[29][1] < 128:
		crop_top_left_x = 250 - 128
		crop_top_right_y = 250  - 128
		cropped = image[crop_top_right_y : crop_top_right_y + 256, crop_top_left_x : crop_top_left_x + 256]
		gray = cv2.cvtColor(cropped,cv2.COLOR_BGR2GRAY)
		cv2.imwrite(name + ".jpg",gray)
		return gray
	ratio = 128/distance
	h1 = int(image.shape[0]*ratio)
	w1 = int(image.shape[1]*ratio)
	dim=(w1,h1)
	resized_img = cv2.resize(img,dim)
	nose_center = (int(shape[29][0]*ratio),int(shape[29][1]*ratio))

	crop_top_left_x = nose_center[0] - 128
	crop_top_right_y = nose_center[1] - 128
	print(crop_top_right_y, crop_top_left_x)
	cropped = resized_img[crop_top_right_y : crop_top_right_y + 256, crop_top_left_x : crop_top_left_x + 256]
	gray = cv2.cvtColor(cropped,cv2.COLOR_BGR2GRAY)
	cv2.imwrite(name + ".jpg",gray)
	return gray
	
#return list of images from a folder
def input(n):
        name = n
        images = []
        for filename in os.listdir(os.getcwd() + "/" + name):
            image = cv2.imread(os.getcwd() + "/" + name + "/" + filename, 1)
            images.append(image)
        return images


directory = "AVR_data/"
name = 0

for x in os.listdir(directory):
		r = input(directory+x)
		os.mkdir("output/" + x)
		name = 0
		for img in r:
			image = preprocess(img, "output/"+ x +"/"+str(name))
			print(x, name)
			name += 1

print("preprocessing done")



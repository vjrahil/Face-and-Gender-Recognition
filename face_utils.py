
import numpy as np
import cv2
import os
import math
import dlib
import imutils


#return the coordinates of a rectangle
def rect_to_bb(rect):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
 
	# return a tuple of (x, y, w, h)
	return (x, y, w, h)

#return a list (68,2) landmarks on the face 
def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)
 
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
 
	# return the list of (x, y)-coordinates
	return coords

#creating 68 points on the face and returning the list of (68,2) landmarks on the face
def points_68(img,n):
	# initialize dlib's face detector (HOG-based) and then create
	# the facial landmark predictor
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor('/home/vj_rahil/Desktop/VR mini proj 1/shape_predictor_68_face_landmarks.dat')
	image = img
	
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	
	# detect faces in the grayscale image
	rects = detector(gray, 1)
	shape = np.zeros((68, 2), dtype=int)
	for rect in rects:
 		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = shape_to_np(shape)
		if n == 1:
			(x, y, w, h) = rect_to_bb(rect)
			cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
			# loop over the (x, y)-coordinates for the facial landmarks
			# and draw them on the image
			for (x, y) in shape:
				cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
	#cv2.imwrite("dlib_1.jpg",image)
	return shape

#calculaing top accuracy
def calculating_top(classifier,X_test,Y_test):
	#calculating top 10,5,3,1
	y_test_predicted_probability = classifier.predict_proba(X_test)
	top = [10,5,3,1]
	#some code to do this
	for ii in range(len(top)):
		probs = classifier.predict_proba(X_test)
		best_n = np.argsort(probs, axis=1)[:,-top[ii]:]
		count=0
		for i in range(len(Y_test)):
			for j in range(len(best_n[0])):
				if Y_test[i]==best_n[i][j]:
					count+=1
					break
		print("Top ",top[ii]," accuracy: ",count/len(Y_test))

#importing libraries
import numpy as np
import pandas as pd
import cv2
import os
import math
import dlib
import imutils
import face_utils
import train_test as tt
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


X_train = []
X_test = []
Y_train = []
Y_test = []
X_train, X_test, Y_train, Y_test = tt.split()

print("Images in X_train = ",len(X_train))
print("Images in X_test = ",len(X_test))

#removing images with dimension other than (256,256,3)
for i in X_train:
	if i.shape != (256,256,3):
		j = X_train.index(i)
		del X_train[j]
		del Y_train[j]


for i in X_test:
	if i.shape != (256,256,3):
		j = X_test.index(i)
		del X_test[j]
		del Y_test[j]

#converting dimension of images into 1-d  
X_train = np.array(X_train) # 4-d
X_tt = X_train.reshape((X_train.shape[0],256*256*3))
print("After conversion dimension for X_train = ", X_tt.shape)
X_test = np.array(X_test)
X_tt1 =X_test.reshape((X_test.shape[0],256*256*3))
print("After conversion dimension for X_train = ", X_tt1.shape)

#PCA model (unsupervised learning)
pca = PCA(n_components = 63,whiten = True)
#Apply PCA transformation(dimension reduction)
X_train_pca = pca.fit_transform(X_tt)
X_test_pca = pca.transform(X_tt1)
#SVC Classifier
classifier = SVC(kernel ='linear', probability=True)
classifier.fit(X_train_pca,Y_train)
#Score
temp = classifier.score(X_test_pca,Y_test)
print("PCA Results = ",temp)
face_utils.calculating_top(classifier,X_test_pca,Y_test)


#LDA model(supervised learning) 
classifier_1 = LDA()  
X_train_lda = classifier_1.fit_transform(X_tt, Y_train)  
print("LDA Results")
face_utils.calculating_top(classifier_1,X_tt1,Y_test)


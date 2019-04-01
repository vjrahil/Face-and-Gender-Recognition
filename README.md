# Face-and-Gender-Recognition

# How to run the code !!
1) Select the dataset on which you want to do face recognition and gender recognition.
2) Create an output folder and run the q.py file which will preprocess all your images and add to the output folder in respective folders.
3) Run the pca.py file which will give you the accuracy of face recogniton based on PCA and LDA.
4) Create output1 folder and create two sub-folders ,naming them male and female and add the male and female folders from the dataset in them respectively.
5) Run the gender.py to give you the accuracy of gender recognition using PCA and SVM. 

# Use of dlib for preprocessing !!
Due to very unreliable result of haar-cascade, we searched for other face recognition library.
We came across ​ dlib​ while searching for face recognition library. We found that the results of
dlib​ were much better than haar-cascade. So we used ​ dlib​ instead of haar-cascade.
The pre-trained facial landmark detector inside the dlib library is used to estimate the location of
68 (x, y)-coordinates that map to facial structures on the face.
These annotations are part of the 68 point ​ iBUG 300-W dataset​ which the dlib facial landmark
predictor was trained on.
Regardless of which dataset is used, the same dlib framework can be leveraged to train a
shape predictor on the input training data — this is useful if you would like to train facial
landmark detectors or custom shape predictors of your own.

To use dlib we rescaled the images to 500*500.
After using dlib on the images and getting coordinates for eyes and nose tip, we carried on with
the rest of the preprocessing steps.
1. For center of the eyes we calculated the midpoint of coordinates​ 37​ and​ 40​ for left eye
and coordinates ​ 43​ and ​ 46​ for right eye.
2. To scale the image such that the distance between eye midpoints is 128 pixels, we
calculated the distance between both eyes and found out the ratio by which image
should be scaled. (ratio = distance/128)
3. We rescaled the image to new dimensions. (new dimension = current dimension * ratio)
4. To get the eyes, nose and lips in a 256*256 frame we take coordinate no.30 as center of
frame and crop the image.
5. At last we converted all the images to grayscale and saved in output folder.

# Use of PCA for dimension reduction !!
The idea behind PCA is that we want to select the hyperplane such that when all the points are
projected onto it, they are maximally spread out. In other words, we want the ​ axis of maximal
variance! Let’s consider our example plot above. A potential axis is the x-axis or y-axis, but, in
both cases, that’s not the best axis. However, if we pick a line that cuts through our data
diagonally, that is the axis where the data would be most spread.After this we applied the
SVM(Linear) classifier giving us the results as follows:
Score = 60.058%

# PCA vs LDA !!
Both Linear Discriminant Analysis (LDA) and Principal Component Analysis (PCA) are
linear transformation techniques that are commonly used for dimensionality reduction.
PCA can be described as an “unsupervised” algorithm, since it “ignores” class labels
and its goal is to find the directions (the so-called principal components) that maximize
the variance in a dataset. In contrast to PCA, LDA is “supervised” and computes the
directions (“linear discriminants”) that will represent the axes that that maximize the
separation between multiple classes.
Score for LDA = 35.58%

# Gender Recognition !!
For gender recogniton we used the same model created for face recognition.
Score  = 85.89%

# Possible solution for Gender Recogniton !!
The first step is to do preprocessing. We are already doing preprocessing for the given dataset
for the face recognition.Now for the recognition algorithm there are some methods by which we
can implement the gender recognition.Gender recognition using OpenCV's fisherfaces
implementation is quite popular. But implementing using CNN will be easy and effective for our
case . For our dataset we need to label each photo as male or female to train the CNN. For that
we can use dictionaries and match each name with male or female . Now we do cross
validation and train the data and get the accuracies and compare for the best model.

# References !!
1)​ https://pythonmachinelearning.pro/face-recognition-with-eigenfaces/ <br />
2)​ https://sebastianraschka.com/Articles/2014_python_lda.html<br />
3)​ https://www.pyimagesearch.com/2018/06/18/face-recognition-with-opencv-python-and-deep-learning/<br />
4)​ https://cmusatyalab.github.io/openface/<br />
5)​ https://datascience.stackexchange.com/questions/18804/sklearn-select-n-best-using-classifier<br />

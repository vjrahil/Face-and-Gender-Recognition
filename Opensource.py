import numpy as np
import pandas as pd
from sklearn.svm import SVC,LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import svm

X = pd.read_csv('features/reps.csv',header=None)
labels = pd.read_csv('features/labels.csv',header=None)

y = labels.iloc[:,0]
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.35, random_state = 33)

Crange = 10.0 ** np.arange(-3, 3)
gamma = 10.0 ** np.arange(-3, 3)
para_grid = dict(gamma = gamma.tolist(), C = Crange.tolist())
model = GridSearchCV(SVC(kernel='rbf'), para_grid, cv = 4, n_jobs = -1)

model.fit(Xtrain,ytrain)
score = model.score(Xtest,ytest)
print(score)


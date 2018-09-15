import numpy as np
from sklearn.neural_network import MLPClassifier
from Load_Data_Blind import *
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV,validation_curve
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import _pickle as cPickle

# load it again
with open('RF_Classifier(Normalized).pkl', 'rb') as fid:
	clf = cPickle.load(fid)

train,test = get_all_data()
train_X,train_Y=train[0],train[1]
test_x,test_y = test[0],test[1]
num_iter = np.dot(list(range(1,20)),50)
BitterPrec = []
NonBitterPrec = []
BitterRecall = []
NonBitterRecall = []
testRes = []

score = clf.predict(test_x)
s = clf.score(test_x,test_y)
testRes.append(s)
# precTrain ,recTrain,_,_ = precision_recall_fscore_support(train_Y,clf.predict(train_X))
precTest ,recTest,_,_ = precision_recall_fscore_support(test_y,score)
BitterPrec.append(precTest[1])
NonBitterPrec.append(precTest[0])
BitterRecall.append(recTest[1])
NonBitterRecall.append(recTest[0])
fpr = 1-recTest[0]
tpr = recTest[1]
print("\n=====================================\n")
print("TPR is {}, FPR is {}, and Test Score is {}".format(tpr,fpr,s))
print("=====================================\n")
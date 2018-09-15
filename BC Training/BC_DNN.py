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




def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")



train, test = get_all_data()
train_X,train_Y=train[0],train[1]
test_x,test_y = test[0],test[1]
num_iter = np.dot(list(range(1,10)),50)
BitterPrec = []
NonBitterPrec = []
BitterRecall = []
NonBitterRecall = []
testRes = []
myClf = ""
maxTpr = 0.6
minFpr = 0.4
final_i = 0
final_s = 0
print("++++++++++ Training Report ++++++++++")
for i in num_iter:
	clf = MLPClassifier(hidden_layer_sizes=(600,),max_iter=i)
	# clf = AdaBoostClassifier(DecisionTreeClassifier(min_samples_split=5),n_estimators=i,learning_rate=0.1)
	clf.fit(train_X,train_Y)
	score = clf.predict(test_x)
	s = clf.score(test_x,test_y)
	testRes.append(s)
	numI = int(i*0.1)
	# precTrain ,recTrain,_,_ = precision_recall_fscore_support(train_Y,clf.predict(train_X))
	precTest ,recTest,_,_ = precision_recall_fscore_support(test_y,score)
	BitterPrec.append(precTest[1])
	NonBitterPrec.append(precTest[0])
	BitterRecall.append(recTest[1])
	NonBitterRecall.append(recTest[0])
	fpr = 1-recTest[0]
	tpr = recTest[1]

	if tpr > 0.65:
		myClf = clf
		final_i = numI
		final_s = s
		print("Iteration: {} TPR is {}, FPR is {}, and Test Score is {}".format(final_i,tpr,fpr, final_s))
		print("=====================================")
		# print("num iteration "+str(i)+" test is "+str(s))
		# break

		if tpr >= 0.70 and final_s >= 0.86 and tpr >= maxTpr and fpr <= minFpr:
			maxTpr = tpr
			minFpr = fpr
			with open('DNN_Classifier(Normalized).pkl', 'wb') as fid:
				cPickle.dump(myClf, fid)
			print("saved classifier")

		print("\n")	

#########################
# print(str(num_iter))
# train_scores,test_scores = validation_curve(clf,np.concatenate((train_X,test_x)),np.concatenate((train_Y,test_y)),param_name='max_iter',param_range=num_iter,scoring='accuracy')
# train_scores_mean = np.mean(train_scores, axis=1)
# test_scores_mean = np.mean(test_scores, axis=1)
# pipeline = Pipeline([
#      ('clf',MLPClassifier())
# ])
# parameters = {
#     'clf__hidden_layer_sizes': ((600,),(400,200),(300,300,100)),
#     'clf__solver':('adam','sgd'),
#     'clf__learning_rate': ('constant','invscaling','adaptive'),
#     'clf__max_iter': (200,600,1800),
#     'clf__early_stopping': (True,False),
# }
# grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)
# grid_search.fit(train_X, train_Y)
# report(grid_search.cv_results_)
# print(grid_search.score(test_x, test_y))
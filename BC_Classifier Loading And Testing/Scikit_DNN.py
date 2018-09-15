import numpy as np
from sklearn.neural_network import MLPClassifier
from Load_Data_Blind import *
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV,validation_curve
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score  as score
import _pickle as cPickle
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier



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



train,test = get_all_data()
train_X,train_Y=train[0],train[1]
test_x,test_y = test[0],test[1]
num_iter = np.dot(list(range(1,25)),10)
BitterPrec = []
NonBitterPrec = []
BitterRecall = []
NonBitterRecall = []
testRes = []
clf = ""
maxTpr = 0.6
minFpr = 0.4
final_i = 0
final_s = 0
ones = np.where(test_y == 1)
sh = np.shape(ones)
ones = np.reshape(ones,[sh[1],sh[0]])
zero = np.where(test_y == 0)
sh = np.shape(zero)
zero = np.reshape(zero,[sh[1],sh[0]])

test_bitter = test_x[ones]
test_nonbitter = test_x[zero]
sh = np.shape(test_bitter)
test_bitter = np.reshape(test_bitter,[sh[0],sh[2]])
sh = np.shape(test_nonbitter)
test_nonbitter = np.reshape(test_nonbitter,[sh[0],sh[2]])

with open('DNN_Classifier(Normalized).pkl','rb') as fid:
    clf = cPickle.load(fid)

myClf = clf


def get_temp_pred(arr,threshold):
    result = []
    # i[0] nonBitter (0) , i[1] bitter (1)
    for i in arr:
        if (max(i[0] ,i[1])) == i[0] and (i[0] - i[1]) >= threshold:
            
            result.append(0)
        else:
            result.append(1)
    return np.array(result)


def check_threshold(myClf,test_bitter,test_nonbitter,test_x,test_y):
    i = [0.0,0.1,0.2,0.3,0.4,0.5]
    test_Bitter_withProb = myClf.predict_proba(test_bitter)
    test_NonBitter_withProb = myClf.predict_proba(test_nonbitter)
    threshold = [j/2 for j in i]
    test_all = myClf.predict_proba(test_x)
    red = []
    blue = []
    green = []
    for s in i :
        temp = get_temp_pred(test_Bitter_withProb,s)
        bitter_acc = score(temp , np.ones(len(temp)))
        red.append(bitter_acc)
        temp2 = get_temp_pred(test_NonBitter_withProb,s)
        nonbitter_acc = score(temp2,np.zeros(len(temp2)))
        blue.append(nonbitter_acc)
        temp3 = get_temp_pred(test_all,s)
        all_acc = score(temp3,test_y)
        green.append(all_acc)
        if i.index(s) == 4:
            print("########### BC_DNN Results ###########")
            print("test Bitter: " + str(bitter_acc))
            print("test NonBitter: " + str(nonbitter_acc))
            print("test overall: " + str(all_acc))
            print("\n===============================\n")
    return red,blue,green,i



red,blue,green,threshold = check_threshold(myClf,test_bitter,test_nonbitter,test_x,test_y)

# red2,blue2,threshold2 = check_threshold(myClf,blind_set_1,blind_set_1)



# print("score first 10 is "+str(score[0:10]))

# print("pred_proba first 10 0: "+str(test_threshold[0:10]) )

# for i in range(0,len(test_threshold)):
#     # Bitter insert to red
#     if i in ones:

# 
# print(str(num_iter))
# train_scores,test_scores = validation_curve(clf,np.concatenate((train_X,test_x)),np.concatenate((train_Y,test_y)),param_name='max_iter',param_range=num_iter,scoring='accuracy')
# train_scores_mean = np.mean(train_scores, axis=1)
# test_scores_mean = np.mean(test_scores, axis=1)

# print("shape shape shape "+str(np.shape(test_threshold)))

# print("lin x1 is {}, len y1 is {} , len x2 is {},len y2 is {}".format(len(list(range(len(ones)))),len(ones),len(list(range(len(test_threshold)-len(ones),len(test_threshold)))),len(zero)))
fig2 = plt.figure()
threshold = [i/2 for i in threshold]
# sub1 = [test_threshold[int(i)][1] for i in ones]
# sub2 = [test_threshold[int(i)][0] for i in ones]
# # bitter -  nonBitter
# sub = np.subtract(sub1,sub2)
# error = np.where(sub < 0)
# print("shape is "+str(np.shape(error)))
# ax = fig2.add_subplot(211)
# print("error bitter "+str(np.shape(error)[1] / len(sub)) + " erros are "+str(len(error)) + " all length is "+str(len(ones)))
# ax.set_title("for Bitter p(Bitter) - p(NonBitter)")
# ax.plot(list(range(len(ones))),list(sub),'r*')
# ax2 = fig2.add_subplot(211)
# sub1 = [test_threshold[int(i)][0] for i in zero]
# sub2 = [test_threshold[int(i)][1] for i in zero]
# sub = np.subtract(sub1,sub2)
plt.title("bitter red, nonbitter blue , all green")
plt.xlabel("threshold")
plt.ylabel("score")
plt.plot(threshold,red,'r-',threshold,blue,'b-',threshold,green,'g-')
# ax3 = fig2.add_subplot(212)
# ax3.set_title("blind1 (bitter) red , blind2(nonbitter) blue")
# ax3.set_xlabel("thresholds")
# ax3.set_ylabel("score")
# ax3.plot(threshold,red2,'r-',threshold,blue2,'b-')

plt.show()
# ax2.plot(list(range(len(zero))),list(sub),'b*')
# # plt.plot(list(range(len(ones))),[test_threshold[int(i)][1] for i in ones],'r*',list(range(len(ones))),[test_threshold[int(j)][0] for j in ones],'b*')
# plt.show()
# fig = plt.figure()
# ax = fig.add_subplot(211)
# # fig.subplots_adjust()
# ax2 = fig.add_subplot(212)

# ax2.set_title("ROC")
# ax2.set_xlabel("FPR")
# ax2.set_ylabel("TPR")
# ax2.plot(newX,newY,'r*')
# ax.set_title("Validation Curve with NeuralNetwork")
# # ax.set_xlabel("num_iteration")
# ax.set_ylabel("Score")
# # ax.ylim(0.0, 1.1)
# ax.plot(num_iter,NonBitterPrec,'r-',num_iter,NonBitterRecall,'b-')
# ax.plot(num_iter,BitterPrec,'y-',num_iter,BitterRecall,'g-')
# # ax.set_xlim([1000,2000])
# fig.text(0.99, 0.89, 'RED NONBITTER Precision\nBLUE NONBITTER Recall\nYellow BITTER Precision\nGreen BITTER Recal',
#         verticalalignment='bottom', horizontalalignment='right',
#         transform=ax.transAxes,
#         fontsize=15)
# plt.show()
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
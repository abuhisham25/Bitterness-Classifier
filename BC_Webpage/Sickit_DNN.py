import numpy as np
from sklearn.neural_network import MLPClassifier
# from Load_Data_Blind import *
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV,validation_curve
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score  as score
from sklearn.preprocessing import StandardScaler
# import cPickle as pickle
# import _pickle as cPickle
import _pickle as pickle
from load_data import *
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


def get_temp_pred(arr,threshold):
    result = []
    # i[0] nonBitter (0) , i[1] bitter (1)
    for i in arr:
        if (max(i[0] ,i[1])) == i[0] and (i[0] - i[1]) >= threshold:
            result.append(0)
        else:
            result.append(1)
    return np.array(result)


def predict_blind(BCThreshold):
    topredict = load_dataset(0,'ToPredict.xlsx')
    topredict[:,58] = topredict[:,59]
    topredict = topredict[:,:59]
    topredict = StandardScaler().fit_transform(topredict)    
    with open('DNN_Classifier(Normalized).pkl','rb') as fid:
        clf = pickle.load(fid)
    with open('RF_Classifier(Normalized).pkl ','rb') as fid2:
        clf_2 = pickle.load(fid2)
        
    myClf = clf

    probResult = myClf.predict_proba(topredict)
    probResult_2 = clf_2.predict_proba(topredict)
    result_dnn = get_temp_pred(probResult,threshold=BCThreshold)
    result_rf = get_temp_pred(probResult_2,threshold=BCThreshold)
    # print(str(result))
    return result_rf,result_dnn
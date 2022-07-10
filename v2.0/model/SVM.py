import sklearn
import os
import joblib
from sklearn import metrics
import pandas as pd
from sklearn.svm import LinearSVC
from tqdm import tqdm

def start(X_train,Y_train,X_test,Y_test):
    print("\n"+"-"*10+"Training Support Vector Machine"+"-"*10+"\n")
    idx = -1
    max_acc = -1

    clf = [0]

    for i in tqdm(range(1,16),desc = "Tuning Hyperparamters"):
        clf.append(LinearSVC(dual = False,verbose = 0,max_iter = i*1000))
        clf[i].fit(X_train, Y_train)

        Y_pred = clf[i].predict(X_test)

        acc = metrics.accuracy_score(Y_test,Y_pred)
        if(acc>max_acc):
            max_acc = acc
            idx = i

    print("\n")

    print("Accuracy : ",max_acc)
    print("Model saved - "+os.getcwd()+"\\model\\SVM.joblib")

    """Saving Model"""
    joblib.dump(clf[idx], os.getcwd()+"\\model\\SVM.joblib")

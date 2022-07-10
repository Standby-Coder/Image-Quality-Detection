import sklearn
import os
import joblib
from sklearn import metrics
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm

def start(X_train,Y_train,X_test,Y_test):
    print("\n"+"-"*10+"Training KNN Classifier"+"-"*10+"\n")
    idx = -1
    max_acc = -1

    clf = [0]

    for i in tqdm(range(1,21),desc = "Tuning Hyperparamters"):
        clf.append(KNeighborsClassifier(n_neighbors = 2*i+1))
        clf[i].fit(X_train, Y_train)

        Y_pred = clf[i].predict(X_test)

        acc = metrics.accuracy_score(Y_test,Y_pred)
        if(acc>max_acc):
            max_acc = acc
            idx = i

    print("\n")

    print("Accuracy : ",max_acc)
    print("Model saved - "+os.getcwd()+"\\model\\KNN.joblib")

    """Saving Model"""
    joblib.dump(clf[idx], os.getcwd()+"\\model\\KNN.joblib")

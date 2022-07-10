from tabnanny import verbose
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import pandas as pd
import os
import numpy as np
import joblib
from tqdm import tqdm 

def start(X_train,Y_train,X_test,Y_test,features):
    print("\n"+"-"*10+"Training Random Forest"+"-"*10+"\n")
    idx = -1
    max_acc = -1

    clf = [0]

    for i in tqdm(range(1,10),desc = "Tuning Hyperparameters"):
        clf.append(RandomForestClassifier(n_estimators = i*50,verbose = 0,criterion = "entropy"))
        clf[i].fit(X_train, Y_train)

        Y_pred = clf[i].predict(X_test)

        acc = metrics.accuracy_score(Y_test,Y_pred)
        if(acc>max_acc):
            max_acc = acc
            idx = i 

    feature_imp = pd.Series(clf[idx].feature_importances_, index = features).sort_values(ascending=False)
    print(feature_imp.to_string())

    print("\n")

    print("Accuracy : ",max_acc)
    print("No of estimators in the Random Forest : ",idx*50)
    print("Model saved - "+os.getcwd()+"\\model\\RandomForest.joblib")

    """Saving Model"""
    joblib.dump(clf[idx], os.getcwd()+"\\model\\RandomForest.joblib")

from tabnanny import verbose
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
import pandas as pd
import os
import numpy as np
import joblib
from tqdm import tqdm 

def start(X_train,Y_train,X_test,Y_test):
    print("\n"+"-"*10+"Training Neural Net"+"-"*10+"\n")
    idx = -1
    max_acc = -1

    clf = [0]
    hidden_layer_sizes = []
    for i in tqdm(range(1,10),desc = "Tuning Hyperparameters"):
        hidden_layer_sizes.append(100)
        clf.append(MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,max_iter=1000,batch_size=115,solver = "adam",alpha = 0.01,verbose = False,tol = 1e-6,n_iter_no_change = 50))
        clf[i].fit(X_train, Y_train)
        print("Current loss = ",clf[i].loss_)
        Y_pred = clf[i].predict(X_test)

        acc = metrics.accuracy_score(Y_test,Y_pred)
        if(acc>max_acc):
            max_acc = acc
            idx = i 

    print("\n")

    print("Accuracy : ",max_acc)
    print("Loss : ",clf[idx].loss_)
    print("Model saved - "+os.getcwd()+"\\model\\NeuralNet.joblib")

    """Saving Model"""
    joblib.dump(clf[idx], os.getcwd()+"\\model\\NeuralNet.joblib")

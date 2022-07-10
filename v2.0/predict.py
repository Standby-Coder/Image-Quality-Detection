import os
import cv2
from feature_utils import *
import pandas as pd
import numpy as np
import feature_creation
import joblib


def get_images():
    test = pd.DataFrame(columns=["Filepath"])
    path = os.getcwd()+"\\test"
    with os.scandir(path) as files:
        for file in files:
            if(os.path.exists(path+"\\"+file.name) and cv2.imread(path+"\\"+file.name) is not None):
                data = {"Filepath" : path+"\\"+file.name}
                p = pd.DataFrame(data,index = [0])
                test = pd.concat([test,p], ignore_index = True, names = ["Filepath"])
    return test

def predict(test):
    features = ["Spatial_Frequency","Histogram Entropy","Squared Gradient","Energy of Gradient","Variance of Laplacian","Brenner's Measure","Wavelet Coefficient sum","Wavelet Coefficient var"]
    X_test = test.loc[:,features].to_numpy()
    result = test.loc[:,"Filepath"].to_frame()
    result = neuralnet(result,X_test)
    result = svm(result,X_test)
    result = knn(result,X_test)
    result = rf(result,X_test)

    result.to_csv(os.getcwd()+"\\test-results\\Results.csv")
    print(result.to_string())

def start():
    test = get_images()
    test = feature_creation.get_features(test)
    predict(test)

def neuralnet(result,X_test):
    nn = joblib.load(os.getcwd()+"\\model\\NeuralNet.joblib")
    Y = nn.predict(X_test)
    result.insert(1,"NeuralNet Predictions",Y,allow_duplicates = True)
    return result

def knn(result,X_test):
    knn = joblib.load(os.getcwd()+"\\model\\KNN.joblib")
    Y = knn.predict(X_test)
    result.insert(1,"KNN Predictions",Y,allow_duplicates = True)
    return result

def svm(result,X_test):
    svm = joblib.load(os.getcwd()+"\\model\\SVM.joblib")
    Y = svm.predict(X_test)
    result.insert(1,"SVM Predictions",Y,allow_duplicates = True)
    return result

def rf(result,X_test):
    rf = joblib.load(os.getcwd()+"\\model\\RandomForest.joblib")
    Y = rf.predict(X_test)
    result.insert(1,"RandomForest Predictions",Y,allow_duplicates = True)
    return result
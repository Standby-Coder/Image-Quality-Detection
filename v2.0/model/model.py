from model import knn
from model import NeuralNet
from model import RandomForest
from model import SVM
import os
import pandas as pd

def start():
    train = pd.read_csv(os.getcwd()+"\\Feature Data\\train.csv")
    train = train.sample(frac=1).reset_index(drop = True)
    test = pd.read_csv(os.getcwd()+"\\Feature Data\\test.csv")

    features = ["Spatial_Frequency","Histogram Entropy","Squared Gradient","Energy of Gradient","Variance of Laplacian","Brenner's Measure","Wavelet Coefficient sum","Wavelet Coefficient var"]

    X_train = train.loc[:,features].to_numpy()
    Y_train = train.loc[:,"Result"].to_numpy()

    X_test = test.loc[:,features].to_numpy()
    Y_test = test.loc[:,"Result"].to_numpy()

    knn.start(X_train,Y_train,X_test,Y_test)
    SVM.start(X_train,Y_train,X_test,Y_test)
    RandomForest.start(X_train,Y_train,X_test,Y_test)
    NeuralNet.start(X_train,Y_train,X_test,Y_test)
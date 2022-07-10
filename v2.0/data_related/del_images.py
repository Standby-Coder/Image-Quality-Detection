import os

def free():
    os.rmdir(os.getcwd()+"\\data")
    os.makedirs(os.getcwd()+"\\data")
    if(os.path.exists(os.getcwd()+"\\temp")): os.rmdir(os.getcwd()+"\\temp")
    if(os.path.exists(os.getcwd()+"\\Feature Data")): os.rmdir(os.getcwd()+"\\Feature Data")
    os.rmdir(os.getcwd()+"\\test")
    os.makedirs(os.getcwd()+"\\test")
    os.rmdir(os.getcwd()+"\\test-results")
    os.makedirs(os.getcwd()+"\\test-results")
    if(os.path.exists(os.getcwd()+"\\model\\KNN.joblib")): os.remove(os.getcwd()+"\\model\\KNN.joblib")
    if(os.path.exists(os.getcwd()+"\\model\\NeuralNet.joblib")): os.remove(os.getcwd()+"\\model\\NeuralNet.joblib")
    if(os.path.exists(os.getcwd()+"\\model\\RandomForest.joblib")): os.remove(os.getcwd()+"\\model\\RandomForest.joblib")
    if(os.path.exists(os.getcwd()+"\\model\\SVM.joblib")): os.remove(os.getcwd()+"\\model\\SVM.joblib")
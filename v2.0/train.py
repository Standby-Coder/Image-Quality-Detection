import os
from data_related.check_data import check_data
import data_related.get_images as get_images
import data_related.data_augment as data_augment
import data_related.data_preprocess as data_preprocess
import data_related.del_images as del_images
from feature_creation import get_features
from model import model
import predict as predict
import pandas as pd

def start():
    """Get the images from internet"""
    print("\n"+"-"*10+"Getting Data"+"-"*10+"\n")
    get_images.start('http://mklab.iti.gr/files/imageblur/CERTH_ImageBlurDataset.zip')
    print("\n"+"-"*10+"Completed data gathering"+"-"*10+"\n")

    """Pre-processing the given images into a dataframe consisting of the filepath and the label"""
    print("\n"+"-"*10+"Data Preprocessing"+"-"*10+"\n")
    dp = data_preprocess.DataPreprocess()
    train = dp.getDataframe()
    test = dp.evalDataframe()
    print("\n"+"-"*10+"Completed Data Preprocessing"+"-"*10+"\n")

    """Checking Data Integrity and removing bad file names"""
    print("\n"+"-"*10+"Checking Data Integrity Step-1"+"-"*10+"\n")
    train = check_data(train)
    print("Training data verified")

    test = check_data(test)
    print("Evaluation Data verified")

    print("\n"+"-"*10+"Data Integrity Check Step-1 Completed"+"-"*10+"\n")

    """Creating more data from given data"""
    print("\n"+"-"*10+"Starting Data Augmentation"+"-"*10+"\n")
    da = data_augment.DataAugmenter(train)
    da.makeFolders()
    train = da.augment()
    print("\n"+"-"*10+"Completed Data Augmentation"+"-"*10+"\n")

    """Checking Data Integrity and removing bad file names"""
    print("\n"+"-"*10+"Checking Data Integrity Step-2"+"-"*10+"\n")
    train = check_data(train)
    print("Training data verified")

    test = check_data(test)
    print("Evaluation Data verified")

    print("\n"+"-"*10+"Data Integrity Check Step-2 Completed"+"-"*10+"\n")

    """Feature Engineering"""
    print("\n"+"-"*10+"Starting Feature Engineering"+"-"*10+"\n")
    train = get_features(train)
    test = get_features(test)
    print("\n"+"-"*10+"Feature Engineering Completed"+"-"*10+"\n")

    """Displaying the final training data"""
    if(not os.path.exists(os.getcwd()+"\\Feature Data")):
        os.makedirs(os.getcwd()+"\\Feature Data")
    train.to_csv(os.getcwd()+"\\Feature Data\\train.csv")
    test.to_csv(os.getcwd()+"\\Feature Data\\test.csv")
    print("\n"+"-"*10+"Training Data"+"-"*10+"\n")
    print(train)
    print("\n"+"-"*10+"Test Data"+"-"*10+"\n")
    print(test)

    print("\n"+"-"*10+"Model Training"+"-"*10+"\n")
    """"Model Training based on Focal Measures and Label"""
    model.start()
    
    print("\n"+"-"*10+"Models Prepared"+"-"*10+"\n")
    # return 1
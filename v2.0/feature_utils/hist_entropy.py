import cv2
import imutils
import math
import numpy as np
from tqdm import tqdm

def hist_entropy(image):
    w,h = image.shape
    tot = w*h
    hist = cv2.calcHist([image],[0],None,[256],[0,256])
    entropy = 0
    for i in range (0,256):
        if hist[i][0] != 0:
            entropy = entropy - (hist[i][0]/tot)*math.log(hist[i][0]/tot)
    return entropy

def getEntropyMeasure(imagePath):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = hist_entropy(gray)
    return fm

def start(train):
    fm = []
    for i in tqdm(range(len(train)), desc = "Histogram Entropy"):
        fm.append(getEntropyMeasure(train.iloc[i]["Filepath"]))
    train.insert(1,"Histogram Entropy",fm,allow_duplicates = True)
    return train
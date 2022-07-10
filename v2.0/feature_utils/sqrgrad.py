import imutils
import cv2
import numpy as np
from tqdm import tqdm

def focal_measure(image):
    ix = np.diff(image, n = 1, axis = 1)
    fm = ix*ix
    fm = np.average(fm)
    return fm

def getFocalMeasure(imagePath):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = focal_measure(gray)
    return fm

def start(train):
    fm = []
    for i in tqdm(range(len(train)), desc = "Squared Gradient"):
        fm.append(getFocalMeasure(train.iloc[i]["Filepath"]))
    train.insert(1,"Squared Gradient",fm,allow_duplicates = True)
    return train
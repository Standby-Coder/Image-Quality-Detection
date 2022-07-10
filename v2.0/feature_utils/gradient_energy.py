import numpy as np
import imutils
import cv2
from tqdm import tqdm

def focal_measure(image):
    ix = image
    iy = image
    iy[:-1,:] = np.diff(image, n = 1, axis = 0)
    ix[:,:-1] = np.diff(image, n = 1, axis = 1)
    fm = ix*ix + iy*iy
    fm = np.average(fm)
    return fm

def getFocalMeasure(imagePath):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = focal_measure(gray)
    return fm

def start(train):
    fm = []
    for i in tqdm(range(len(train)), desc = "Energy of Gradient"):
        fm.append(getFocalMeasure(train.iloc[i]["Filepath"]))
    train.insert(1,"Energy of Gradient",fm,allow_duplicates = True)
    return train
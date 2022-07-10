import imutils
import cv2
import numpy as np
from tqdm import tqdm

def focal_measure(image):
    m,n = image.shape
    dh = np.zeros((m,n))
    dv = np.zeros((m,n))
    dv[:m-2,:] = image[2:,:] - image[:-2,:]
    dh[:,:n-2] = image[:,2:] - image[:,:-2]
    fm = np.maximum(dh,dv)
    fm = fm*fm
    fm = np.average(fm)
    return fm

def getFocalMeasure(imagePath):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = focal_measure(gray)
    return fm

def start(train):
    fm = []
    for i in tqdm(range(len(train)), desc = "Brenner's Measure"):
        fm.append(getFocalMeasure(train.iloc[i]["Filepath"]))
    train.insert(1,"Brenner's Measure",fm,allow_duplicates = True)
    return train
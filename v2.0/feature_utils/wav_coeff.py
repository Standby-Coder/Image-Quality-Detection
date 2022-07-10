import numpy as np
import imutils
import cv2
import pywt
from tqdm import tqdm

def focal_measure(image,mode):
    coeffs = pywt.dwt2(image,'db6')
    _,(H,V,D) = coeffs
    if(mode == 'sum'):
        fm = abs(H)+abs(V)+abs(D)
        fm = np.average(fm)
        return fm
    elif (mode == 'var'):
        fm = np.var(abs(H))+np.var(abs(V))+np.var(abs(D))
        return fm


def getFocalMeasure(imagePath,mode):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = focal_measure(gray,mode)
    return fm

def start(train,mode):
    fm = []
    for i in tqdm(range(len(train)), desc = "Wavelet Coefficient "+mode):
        fm.append(getFocalMeasure(train.iloc[i]["Filepath"],mode))
    train.insert(1,"Wavelet Coefficient "+mode,fm,allow_duplicates = True)
    return train
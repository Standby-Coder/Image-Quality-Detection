import imutils
import cv2
from tqdm import tqdm

def focal_measure(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

def getFocalMeasure(imagePath):
    image = cv2.imread(imagePath)
    if(image is not None):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        return None
    fm = focal_measure(gray)
    return fm

def start(train):
    fm = []
    for i in tqdm(range(len(train)), desc = "Variance of Laplacian"):
        fm.append(getFocalMeasure(train.iloc[i]["Filepath"]))
    train.insert(1,"Variance of Laplacian",fm,allow_duplicates = True)
    for i in range(len(train)):
        if(train.iloc[i,1] == None):
            train.drop(i, inplace = True)
    return train
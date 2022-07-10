import os
from os import path
import pandas as pd
from tqdm import tqdm
import cv2

def check_data(traindf):
    l = len(traindf)
    d = []
    for i in tqdm(range(l), desc = "Checking..."):
      if(not path.exists(r""+traindf.iloc[i,0])):
        d.append(i)
      img = cv2.imread(traindf.iloc[i,0])
      if(img is None):
        d.append(i)
    d.reverse()
    for i in d:
      traindf.drop(i, axis = 0, inplace = True)
    return traindf
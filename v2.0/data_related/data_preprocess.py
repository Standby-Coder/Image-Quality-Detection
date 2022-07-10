import os
from os import path
import pandas as pd

class DataPreprocess():
  def __init__(self):
    self.commontrain = os.getcwd()+"\\data\\CERTH_ImageBlurDataset\\TrainingSet"
    self.commontest = os.getcwd()+"\\data\\CERTH_ImageBlurDataset\\EvaluationSet"
    self.traindf = pd.DataFrame(columns=["Filepath","Result"])
    self.testdf = pd.DataFrame(columns = ["Filepath","Result"])
  
  def getDataframe(self):
    with os.scandir(self.commontrain) as entries1:
      for entry1 in entries1:
        if entry1.is_dir():
          if(entry1.name == "Undistorted"):
            with os.scandir(self.commontrain+"\\"+entry1.name) as entries2:
              for entry2 in entries2:
                if entry2.is_file():
                  data = {"Filepath" : self.commontrain+"\\"+entry1.name+"\\"+entry2.name ,"label" : entry1.name ,"Result" : 0}
                  p = pd.DataFrame(data,index = [0])
                  self.traindf = pd.concat([self.traindf,p], ignore_index = True, names = ["Filepath","label","Result"])
          else:
            with os.scandir(self.commontrain+"\\"+entry1.name) as entries3:
              for entry2 in entries3:
                if entry2.is_file():
                  data = {"Filepath" : self.commontrain+"\\"+entry1.name+"\\"+entry2.name ,"label" : entry1.name ,"Result" : 1}
                  p = pd.DataFrame(data,index = [0])
                  self.traindf = pd.concat([self.traindf,p], ignore_index = True, names = ["Filepath","label","Result"]) 
    return self.traindf

  def evalDataframe(self):
    df1 = pd.read_excel(self.commontest+"\\DigitalBlurSet.xlsx",names = ["Filepath","Result"])
    df2 = pd.read_excel(self.commontest+"\\NaturalBlurSet.xlsx",names = ["Filepath","Result"])
    df1["Filepath"] = self.commontest +"\\DigitalBlurSet\\" + df1["Filepath"]
    df2["Filepath"] = self.commontest +"\\NaturalBlurSet\\" + df2["Filepath"] +".jpg"
    self.testdf = pd.concat([df1,df2], ignore_index = True)
    return self.testdf.replace([-1],[0])
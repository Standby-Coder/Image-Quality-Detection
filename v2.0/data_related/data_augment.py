from PIL import Image
import pandas as pd
import os
from os import path
from tqdm import tqdm

class DataAugmenter():
  def __init__(self,train):
    self.traindf = train
    self.path = os.getcwd()+"\\temp\\"
  def makeFolders(self):
    if(not path.exists(os.getcwd()+"\\temp")):
      os.makedirs(os.getcwd()+"\\temp")
      for i in self.traindf.label.unique():
          os.mkdir(os.getcwd()+"\\temp\\"+i)
  def augment(self):
    lst1 = []
    lst2 = []
    lst3 = []
    for i in tqdm(range(self.traindf.shape[0]),desc = "Data Augmenting..."):
      src = self.traindf.iloc[i,0]
      label  = self.traindf.iloc[i,2]
      label_enc = self.traindf.iloc[i,1]
      img = Image.open(src).convert('RGB')
      # img1 = img.transpose(Image.FLIP_LEFT_RIGHT)
      # newpath = self.path+ str(label) +"\\rev"+str(i)+".jpg"
      lst1,lst2,lst3 = self.mirror(i,img,label,label_enc,lst1,lst2,lst3)
      lst1,lst2,lst3 = self.crop(i,img,label,label_enc,lst1,lst2,lst3)
      # lst1.append(newpath)
      # lst2.append(label)
      # lst3.append(label_enc)
      # img1.save(newpath)
    df = pd.DataFrame(list(zip(lst1,lst2,lst3)),columns = ['Filepath','label','Result'])
    self.traindf = pd.concat([df,self.traindf],ignore_index = True,names = ['Filepath','label','Result'])
    return self.traindf

  def mirror(self,i,img,label,label_enc,lst1,lst2,lst3):
    img1 = img.transpose(Image.FLIP_LEFT_RIGHT)
    newpath = self.path+ str(label) +"\\rev"+str(i)+".jpg"
    lst1.append(newpath)
    lst2.append(label)
    lst3.append(label_enc)
    img1.save(newpath)
    return lst1,lst2,lst3

  def crop(self,i,img,label,label_enc,lst1,lst2,lst3):
    imgt = img.transpose(Image.FLIP_LEFT_RIGHT)
    w,h = img.size
    l = 0
    u = 0
    r = w/2
    d = h/2
    l1 = []
    l2 = []
    l3 = []
    img_arr = []
    img_arr.append(img.crop((l,u,r,d)))
    img_arr.append(img.crop((r,u,w,d)))
    img_arr.append(img.crop((l,d,r,h)))
    img_arr.append(img.crop((r,d,w,h)))
    img_arr.append(imgt.crop((l,u,r,d)))
    img_arr.append(imgt.crop((r,u,w,d)))
    img_arr.append(imgt.crop((l,d,r,h)))
    img_arr.append(imgt.crop((r,d,w,h)))
    for j in range(8):
      if j<4 :
        newpath = self.path+ str(label) +"\\"+str(j)+"_crop"+str(i)+".jpg"
      else: 
        newpath = self.path+ str(label) +"\\"+str(j)+"_revcrop"+str(i)+".jpg"
      l1.append(newpath)
      l2.append(label)
      l3.append(label_enc)
      img_arr[j].save(newpath)
    lst1.extend(l1)
    lst2.extend(l2)
    lst3.extend(l3)
    return lst1,lst2,lst3

#   def rotate(self):
#     lst1 = []
#     lst2 = []
#     lst3 = []
#     for i in range(self.traindf.shape[0]):
#       src = self.traindf.iloc[i,0]
#       label  = self.traindf.iloc[i,2]
#       label_enc = self.traindf.iloc[i,1]
#       img = Image.open(src).convert('RGB')
#       img1 = img.transpose(Image.FLIP_LEFT_RIGHT)
#       newpath = self.path+ str(label) +"\\rev"+str(i)+".jpg"
#       lst1.append(newpath)
#       lst2.append(label)
#       lst3.append(label_enc)
#       img1.save(newpath)
#     df = pd.DataFrame(list(zip(lst1,lst2,lst3)),columns = ['Filepath','label','Result'])
#     self.traindf = pd.concat([df,self.traindf],ignore_index = True,namedtuple_sign_logabsdet = ['Filepath','label','Result'])
#     return self.traindf
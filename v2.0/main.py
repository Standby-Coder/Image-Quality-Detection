import get_images
import data_augment
import data_preprocess
import del_images
import model
import predict
import os

os.system("python -u requirements.py")

get_images.start('http://mklab.iti.gr/files/imageblur/CERTH_ImageBlurDataset.zip')

i = input("\n\nEnter q to quit")

if i == 'q': del_images.free()
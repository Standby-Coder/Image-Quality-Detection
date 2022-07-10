import train
import predict
import os
from data_related.del_images import free

def clear():
    if(os.name == 'posix'):
        _ = os.system('clear')
    else:
        _ = os.system('cls')

clear()

print("\n\n"+"*"*20+"Image Blur Detection Tool v2.0"+"*"*20+"\n")
while(True):
    i = input("Enter choice\n1) Train the model on a specified Dataset\n2) Predict whether your images are blurry or not\nPress 'q' to exit\nPlace your images in \"\\test\" folder for option 2\nYour Choice : ")

    if(i=="1"):
        stat = train.start()
    elif(i=="2"):
        predict.start()
    elif(i=="q"):
        free()
        exit()

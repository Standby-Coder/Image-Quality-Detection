import wget
from zipfile import ZipFile
import os
from os import path
import patoolib

cwd = os.getcwd()


# def start(url):
#     site_url = url
#     if(path.exists(os.getcwd()+"\\data")):
#       print("Data Already Downloaded")
#     else:
#       os.makedirs(os.getcwd()+"\\data")
#       file_name = wget.download(site_url,out = os.getcwd()+"\\data")
#       print("File Downloaded - "+file_name)
#     c = 0
#     print("Total Number of Files extracted = "+str(unzip(os.getcwd()+"\\data",c)))

def start(url):
    file_name = "CERTH_ImageBlurDataset.zip"
    c = 0
    if(path.exists(os.getcwd()+"\\data")):
      print("Zip Already Downloaded")
    else:
      os.makedirs(os.getcwd()+"\\data")
    print("Total Number of Files extracted = "+str(unzip(os.getcwd()+"\\data",c)))


def unzip (path, total_count):
    for root, dirs, files in os.walk(path):
        for file in files:
            file_name = os.path.join(root, file)
            if (not (file_name.endswith('.zip') or file_name.endswith('.rar'))):
                total_count += 1
            else:
                currentdir = file_name[:-4]
                # if not os.path.exists(currentdir):
                #     os.makedirs(currentdir)
                if(file_name.endswith('.zip')):
                    with ZipFile(file_name) as zipObj:
                        zipObj.extractall(path)
                elif(file_name.endswith('.rar')):
                    patoolib.extract_archive(file_name,outdir = root)
                os.remove(file_name)
                total_count = unzip(currentdir, total_count)
    return total_count

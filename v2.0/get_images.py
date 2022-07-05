import wget
from zipfile import ZipFile
import os

cwd = os.getcwd()


def start(url):
    site_url = url
    file_name = wget.download(site_url,out = os.getcwd()+"/temp")
    print("File Downloaded - "+file_name)


def unzip (path, total_count):
    for root, dirs, files in os.walk(path):
        for file in files:
            file_name = os.path.join(root, file)
            if (not file_name.endswith('.zip')):
                total_count += 1
            else:
                currentdir = file_name[:-4]
                if not os.path.exists(currentdir):
                    os.makedirs(currentdir)
                with ZipFile(file_name) as zipObj:
                    zipObj.extractall(currentdir)
                os.remove(file_name)
                total_count = unzip(currentdir, total_count)
    return total_count

c = 0
c = unzip(os.getcwd()+"/temp",c)
print("Total Number of Files extracted = "+str(c))

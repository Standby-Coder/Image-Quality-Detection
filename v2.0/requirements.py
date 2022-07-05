import subprocess
import sys
import warnings
warnings.filterwarnings("ignore")

subprocess.check_call([sys.executable,"-m","pip","install","--upgrade","pip"])

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

packages = ["opencv-python", "imutils", "argparse", "torch", "torchmetrics", "numpy", "pandas", "wget"]

for package in packages:
    install(package = package)
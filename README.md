# Image Quality Detection

## v2.0

The given folder contains a bunch of python scripts which covers the entire implementation

**This requires the computer to have 7zip as there are RAR files to be extracted**

Steps to get the program running - 

Step 1 - Install the required python packages listed in `requirements.txt` using pip

```
pip install -r requirements.txt
```

Step 2 - Run the `main.py` script

```
python main.py
```

Implementation Technique - Focal Measure Estimates are calculated, then those are run through some models

Focal Measure Estimates include -

- Variance of Laplacian
- Energy of Gradient
- Squared Gradient
- Histogram Entropy
- Brenner's Measure
- Sum of Wavelet Coefficients
- Variance of Wavelet Coefficients

Models through which the focal measure estimates are fed to

- K-Neighbors Classifier
  - Accuracy - 80%
- Support Vector Machine
  - Accuracy - 80%
- Neural Network (with each hidden layer of 100 neurons)
  - Accuracy - 84%
- Random Forest Classifier
  - Accuracy - 84.7%

List to be implemented/Things which have to be improved

- [ ] Come up with a better version of this markdown
- [ ] Implement more focal measure estimates like DCT Energy Ratio, and such, and to improve accuracy
- [ ] Ensemble the current models and then in future implement more models and add to this ensemble
- [ ] Reduce the number of times images are opened in `feature_creation.py`
- [ ] \(Optional) Implement Multi-Threading in `feature_creation.py` (:thinking::thinking::thinking:)

## v1.0

Hi there!

Given is the Jupyter Notebook (or more preferrably a Google Colab Notebook) which can be deployed 
anywhere irrespective of whether the system has a GPU, TPU or solely runs on CPU. 

Steps to get the notebook up and running - 

Step 1 - Download the dataset of you are working on Jupyter Notebook on Windows. If you are using Kaggle Notebooks/Google 
        Colaboratory/Jupyter Notebooks on Linux then there is no need to download as "wget" will take care of it
	 
	Link to Dataset - "http://mklab.iti.gr/files/imageblur/CERTH_ImageBlurDataset.zip"

Step 2 - Open the notebook. The libraries required are Numpy, Pandas, Seaborn, Matplotlib, OS, Scikit-Learn,
	   	PIL, OpenCV, and PyTorch.
	   	Some of these libraries are already included as part of the Python Installation. Others such as 
        PyTorch needs to be installed seperately.We can use "pip" command to get these libraries

    `!pip install "torchmetrics" "matplotlib" "torch" "seaborn" "transformers" "opencv-python" "imutils" "torchvision" "Pillow"` 

Step 3 - You need to update the filepath of the dataset according to where the wget command
        downloads the dataset.

Step 4 - Then run the cells sequentially.

Step 5 - If you want to apply transfer learning, then after the neural network is trained, a copy of the 
	   	entire model's parameters along with its architecture can be later imported for similar problems 
	   	and can be used to reduce the computation time

Implementation Technique - Images fed into a 3-layer convolutional Neural Network

Accuracy Achieved - 74%

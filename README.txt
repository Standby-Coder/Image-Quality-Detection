Image Quality Detection v1.0

Hi there!

Given is the Jupyter Notebook (or more preferrably a Google Colab Notebook) which can be deployed 
anywhere irrespective of whether the system has a GPU, TPU or solely runs on CPU. 

Steps to get the notebook up and running - 

Step 1 - Download the dataset of you are working on Jupyter Notebook on Windows. If you are using Kaggle Notebooks/Google 
         Colaboratory/Jupyter Notebooks on Linux then there is no need to download as "wget" will take care of it
	 
	 !wget "http://mklab.iti.gr/files/imageblur/CERTH_ImageBlurDataset.zip"

Step 2 - Open the notebook. The libraries required are Numpy, Pandas, Seaborn, Matplotlib, OS, Scikit-Learn,
	   PIL, OpenCV, and PyTorch.
	   Some of these libraries are already included as part of the Python Installation. Others such as 
         PyTorch needs to be installed seperately.We can use "pip" command to get these libraries

         !pip install "torchmetrics" "matplotlib" "torch" "seaborn" "transformers" "opencv-python" "imutils" "torchvision" "Pillow" 

Step 3 - You need to update the filepath of the dataset according to where the wget command
         downloads the dataset.

Step 4 - Then run the cells sequentially.

Step 5 - If you want to apply transfer learning, then after the neural network is trained, a copy of the 
	   entire model's parameters along with its architecture can be later imported for similar problems 
	   and can be used to reduce the computation time

	Accuracy(Training)      Accuracy(Val/test dataset)	Inference Time 

CPU       91.304348			  73.043478 			64 min

GPU	    93.043478                   69.565217                   43 min

TPU       95.458937                   73.913043                   57 min

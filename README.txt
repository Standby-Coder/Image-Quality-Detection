Image Quality Detection v1.0

Hi there!

Given is the Jupyter Notebook (or more preferrably a Google Colab Notebook) which can be deployed 
anywhere irrespective of whether the system has a GPU, TPU or solely runs on CPU. 

Steps to get the notebook up and running - 

Step 1 - Open the notebook. The libraries required are Numpy, Pandas, Seaborn, Matplotlib, OS, Scikit-Learn,
	   PIL, OpenCV, and PyTorch.
	   Some of these libraries are already included as part of the Python Installation. Others such as 
         PyTorch needs to be installed seperately.We can use "pip" command to get these libraries

         !pip install "torchmetrics" "matplotlib" "torch" "seaborn" "transformers" "opencv-python" "imutils" "torchvision" "Pillow" 

Step 2 - You need to update the filepath of the dataset according to where the wget command
         downloads the dataset.

Step 3 - Then run the cells sequentially.

Step 4 - If you want to apply transfer learning, then after the neural network is trained, a copy of the 
	   entire model's parameters along with its architecture can be later imported for similar problems 
	   and can be used to reduce the computation time

	Accuracy(Training)      Accuracy(Val/test dataset)	Inference Time 

CPU       91.304348			  73.043478 			64 min

GPU	    93.043478                   69.565217                   43 min

TPU       95.458937                   73.913043                   57 min
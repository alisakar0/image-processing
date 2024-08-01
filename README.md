This program aims to obtain similar products by performing visual searches on shopping sites.
In our working logic, we train hundreds of products with the Resnet50 model and record each product's own score in the Numpy array.
Then, we plan to obtain the most similar products by comparing the product entered by the user with cosine similarity.
This study was written and run on Googlecolab.

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import torch
import torchvision.transforms as transforms
from IPython.display import display, Image as IPImage
from google.colab import files 

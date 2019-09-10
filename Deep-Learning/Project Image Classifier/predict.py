import argparse
import pandas as pd
import numpy as np
import seaborn as sns

import torch
from torch import nn
from torch import optim
from torchvision import datasets, models, transforms
import torch.nn.functional as F
import torch.utils.data 

from collections import OrderedDict
from PIL import Image

import functions as func
import json


import matplotlib.pyplot as plt
'exec(%matplotlib inline)'

arg = argparse.ArgumentParser(description='Predict.py')

# Aruguments passed through Command line
arg.add_argument('--data_dir', default="./flowers/", help='dataset directory')
arg.add_argument('--image_dir', default="./flowers/test/37/image_03811.jpg", help='dataset directory')
arg.add_argument('--load_dir', default="./chaitanya_imageClassifier_script_checkpoint.pth", help='Enter location to save checkpoint')
arg.add_argument('--top_k', type=int, default=3, help='Enter number of top most likely classes to view, default is 3')
arg.add_argument('--cat_to_name_json_file', default='cat_to_name.json', help='Enter path to image')
arg.add_argument('--gpu', default="cpu", help='Turn GPU mode on or off, default is "cpu" for prediction')

# parsing the argument values 
args = arg.parse_args()

data_dir = args.data_dir
image_dir = args.image_dir
load_dir = args.load_dir
top_k = args.top_k
cat_to_name_json_file = args.cat_to_name_json_file
gpu = args.gpu

train_data_loader, test_data_loader, valid_data_loader, trainData, testData, validData = func.load_data(data_dir)

#mapping from category label to category name
with open(cat_to_name_json_file, 'r') as f:
    cat_to_name = json.load(f)
    

#calculating probabilities and classes
probs, classes = func.predict(image_dir, load_dir, gpu, top_k)


#class_names = [cat_to_name [item] for item in classes]

print('Predicted Classes: ', classes)
print('-----------------------------------------')
print ('Class Names: ')
[print(cat_to_name[x]) for x in classes]
print('-----------------------------------------')
print('Predicted Probability: ', probs)


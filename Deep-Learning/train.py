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
import matplotlib.pyplot as plt
'exec(%matplotlib inline)' 


arg = argparse.ArgumentParser(description='Train.py')

# Aruguments passed through Command line
arg.add_argument('--data_dir', default="./flowers/", help='dataset directory')
arg.add_argument('--save_dir', default="./chaitanya_imageClassifier_script_checkpoint.pth", help='Enter location to save checkpoint')
arg.add_argument('--dropout', default = 0.2, help='Enter dropout for training the model, default is 0.2')
arg.add_argument('--epochs', type=int, default=4, help='Number of epochs for training as int')
arg.add_argument('--learning_rate', default=0.001, help='Define gradient descent learning rate as float')
arg.add_argument('--hidden_units', type=int, default=4096, help='Hidden units for DNN classifier as int')
arg.add_argument('--gpu', default="gpu", help='Turn GPU mode on or off, default is "gpu"')

# parsing the argument values 
args = arg.parse_args()

data_dir=args.data_dir
save_dir=args.save_dir
dropout=args.dropout
epochs=args.epochs
lr=args.learning_rate
hidden_units=args.hidden_units
gpu=args.gpu


train_data_loader, test_data_loader, valid_data_loader, trainData, testData, validData = func.load_data(data_dir)

model, loss_function, optimizer = func.build_image_classifier(hidden_units, dropout, gpu, lr)

model, optimizer = func.train(model, epochs, train_data_loader, test_data_loader, 
                              valid_data_loader, loss_function, optimizer, gpu, 25)

func.save_model(model, trainData, save_dir)

print("--------------------------- Model trained successfully ---------------------------")
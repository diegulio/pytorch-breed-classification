import torch
from torch import nn
from torch.utils.data import Dataset
import pandas as pd
import os
from torchvision.io import read_image
import matplotlib.pyplot as plt
from PIL import Image
import random
import torchvision.transforms as transforms
import lightning as L
import timm
import torch.nn.functional as F
from torch import optim
import wandb
from config import *
from pytorch_lightning.loggers import WandbLogger
from torchmetrics import Accuracy, F1Score
from tqdm.notebook import tqdm
from torchvision.transforms import v2
from torch.optim.lr_scheduler import ReduceLROnPlateau
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from collections import OrderedDict
from sklearn.model_selection import train_test_split


# Labels

class Labels:
    """
    Read labels csv and split dataset into train, val and test
    """
    def __init__(self, labels_path, test_size, val_size, random_state):
        self.labels = pd.read_csv(labels_path)
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        
    def train_test_val_split(self):
        X = labels.id
        y = labels.breed
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state, shuffle = True, stratify = y)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=self.val_size, random_state=self.random_state, shuffle = True, stratify = y_train)
       
        train_data = pd.concat([X_train, y_train], axis = 1).reset_index(drop = True)
        val_data = pd.concat([X_val, y_val], axis = 1).reset_index(drop = True)
        test_data = pd.concat([X_test, y_test], axis = 1).reset_index(drop = True) 
        
        # Clean 4 channel image. TODO: clean
        test_data = test_data[test_data.id!='ca6'].reset_index(drop = True)
        
        return train_data, val_data, test_data
    
class PetDataSet(Dataset):
    
    """
    Pet dataset
    """
    def __init__(self, config, labels, transform):
        self.labels = labels
        self.dir = config.IMG_PATH
        self.transform = transform
        self.config = config

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        breed = self.labels.iloc[idx, 1] # breed 
        class_id = self.config.class_to_idx[breed] # to idx
        img_path = self.labels.iloc[idx, 0] # image path
        full_path = os.path.join(self.dir, f'{img_path}.jpg') # full path
        image = read_image(full_path)/255 # read image and normalize
        img = self.transform(image) # apply transforms
        
        return img, class_id
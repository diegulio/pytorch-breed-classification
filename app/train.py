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
from config import CFG
from pytorch_lightning.loggers import WandbLogger
from torchmetrics import Accuracy, F1Score
from tqdm.notebook import tqdm
from torchvision.transforms import v2
from torch.optim.lr_scheduler import ReduceLROnPlateau
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from collections import OrderedDict
from dataset import PetDataSet, Labels
from torch.utils.data import DataLoader
from backbone import Backbone
from model import PetClassificationModel


def freeze_pretrained_layers(model, model_name):
    '''Freeze all layers except the last layer(fc or classifier)'''
    for param in model.parameters():
            param.requires_grad = False

    # Unfreeze Classifier Parameters EfficientNET
    if model_name == 'efficientnet_b2':
      model.pretrained_model.classifier.weight.requires_grad = True
      model.pretrained_model.classifier.bias.requires_grad = True
    # Unfreeze Classifier Parameters VGG19
    elif model_name == 'vgg19_bn':
      model.pretrained_model.head.fc.weight.requires_grad = True
      model.pretrained_model.head.fc.bias.requires_grad = True
    elif model_name == 'inception_v4':
      model.pretrained_model.last_linear.weight.requires_grad = True
      model.pretrained_model.last_linear.bias.requires_grad = True
    elif model_name == 'resnet50':
      model.pretrained_model.fc.weight.requires_grad = True
      model.pretrained_model.fc.bias.requires_grad = True
    else:
      raise Exception('Modelo no encontrado')



# Load dataset

## Augmentations
if CFG.AUGMENTATION:
  train_transform = v2.Compose([
      v2.Resize(CFG.IMG_SIZE),
      v2.RandomHorizontalFlip(0.4),
      v2.RandomVerticalFlip(0.1),
      v2.RandomApply(transforms=[v2.RandomRotation(degrees=(0, 90))], p=0.5),
      v2.RandomApply(transforms=[v2.ColorJitter(brightness=.3, hue=.1)], p=0.3),
      v2.RandomApply(transforms=[v2.GaussianBlur(kernel_size=(5, 9))], p=0.3),
      #transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
  ])
else:
  train_transform = v2.Compose([
    v2.Resize(CFG.IMG_SIZE),
])

test_transform = v2.Compose([
    v2.Resize(CFG.IMG_SIZE),
])


label_data = Labels(labels_path = CFG.LABEL_PATH, test_size = CFG.TEST_SIZE,val_size =  CFG.VAL_SIZE, random_state = CFG.SEED)
train_data, val_data, test_data = label_data.train_val_test_split()
# Pytorch Datasets
train_dataset = PetDataSet(config = CFG, labels = train_data, transform = train_transform)
val_dataset = PetDataSet(config = CFG, labels = val_data, transform = test_transform)
test_dataset = PetDataSet(config = CFG, labels = test_data, transform = test_transform)


# DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=CFG.BATCH_SIZE, shuffle=True, num_workers = 1)
val_dataloader = DataLoader(val_dataset, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers = 1)
test_dataloader = DataLoader(test_dataset, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers = 1)

# Model
backbone = Backbone(CFG.MODEL, len(CFG.idx_to_class), pretrained = CFG.PRETRAINED)
model = PetClassificationModel(backbone.model, CFG)

# Freeze layers
freeze_pretrained_layers(model, model_name = CFG.MODEL)

# Trainer
wandb_config = {
    'lr' : CFG.LEARNING_RATE,
    'model_name': CFG.MODEL,
    'pretrained': CFG.PRETRAINED,
    'precision': CFG.PRECISION,
    'min_epochs': CFG.MIN_EPOCHS,
    'max_epochs': CFG.MAX_EPOCHS,
    'accelerator': CFG.ACCELERATOR,
    'AUGMENTATION': CFG.AUGMENTATION
}

early_stopping = EarlyStopping(monitor="val_loss", patience=10, verbose=False, mode="min")
callbacks = [early_stopping]

## Logger
wandb_logger = WandbLogger(project = CFG.WANDB_PROJECT,
                           entity = CFG.WANDB_ENTITY,
                           name = f"{CFG.MODEL}_+layers",
                           log_model=False,
                           config = wandb_config,
                           group = 'pretrained',
                           job_type = 'training')

## Train
trainer = L.Trainer(
    accelerator=CFG.ACCELERATOR,
    devices=1,
    min_epochs=CFG.MIN_EPOCHS,
    max_epochs=CFG.MAX_EPOCHS,
    precision=CFG.PRECISION,
    logger = wandb_logger,
   # callbacks = callbacks
)

trainer.fit(model, train_dataloader, val_dataloader)

# Save model
torch.save(model.state_dict(), CFG.MODEL_PATH)
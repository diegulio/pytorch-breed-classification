import pandas as pd

IMG_PATH = '/content/drive/MyDrive/Colab Data/buscomiperro/images'
LABEL_PATH = '/content/drive/MyDrive/Colab Data/buscomiperro/labels.csv'

labels = pd.read_csv(LABEL_PATH)
idx_to_class = dict(enumerate(labels.breed.unique()))
class_to_idx = {c:i for i,c in idx_to_class.items()}


# Model related
LEARNING_RATE = 0.001
MODEL = 'inception_v4'
PRETRAINED = True
PRECISION = 16
MIN_EPOCHS = 1
MAX_EPOCHS = 20
ACCELERATOR = 'gpu'
AUGMENTATION = True

# Wandb related
WANDB_PROJECT = 'breed-classification-pytorch'
WANDB_ENTITY = 'diegulio'
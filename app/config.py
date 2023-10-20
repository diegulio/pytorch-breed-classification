import pandas as pd


class CFG:
    SEED = 13

    IMG_PATH = "/content/drive/MyDrive/Colab Data/buscomiperro/images"
    LABEL_PATH = "data/labels.csv"

    labels = pd.read_csv(LABEL_PATH)
    idx_to_class = dict(enumerate(labels.breed.unique()))
    class_to_idx = {c: i for i, c in idx_to_class.items()}

    # Model related
    LEARNING_RATE = 0.001
    MODEL = "inception_v4"
    PRETRAINED = True
    PRECISION = 16
    MIN_EPOCHS = 1
    MAX_EPOCHS = 20
    ACCELERATOR = "gpu"
    AUGMENTATION = True
    MODEL_PATH = "model.pt"

    # Data
    TEST_SIZE = 0.2
    VAL_SIZE = 0.1
    BATCH_SIZE = 64
    IMG_SIZE = (299, 299)  # Depends in base model

    # Wandb related
    WANDB_PROJECT = "breed-classification-pytorch"
    WANDB_ENTITY = "diegulio"

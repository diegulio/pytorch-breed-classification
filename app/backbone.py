import timm
from dataclasses import dataclass

class Backbone:
    def __init__(self, model, num_classes, pretrained = True):
        self.model = timm.create_model(model, pretrained = pretrained, num_classes = num_classes)
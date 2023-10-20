
import torch
from torch import nn


import lightning as L

import torch.nn.functional as F
from torch import optim
from torchmetrics import Accuracy

from torch.optim.lr_scheduler import ReduceLROnPlateau



class PetClassificationModel(L.LightningModule):
  def __init__(self, base_model, config):
    super().__init__()
    self.config = config
    self.num_classes = len(self.config.idx_to_class)
    metric = Accuracy(task="multiclass", num_classes=self.num_classes)
    self.train_acc = metric.clone()
    self.val_acc = metric.clone()
    self.test_acc = metric.clone()
    self.training_step_outputs = []
    self.validation_step_outputs = []
    self.test_step_outputs = []

    self.pretrained_model = base_model
    out_features = self.pretrained_model.get_classifier().out_features
    self.custom_layers = nn.Sequential(
          nn.Linear(out_features, 512, device = "cuda"),
          nn.ReLU(),
          nn.Dropout(),
          nn.Linear(512, self.num_classes, device = "cuda"),
        )

  def forward(self, x):
    x = self.pretrained_model(x)
    #x = self.custom_layers(x)
    return x


  def training_step(self, batch, batch_idx):
    x,y = batch
    logits = self.forward(x) # -> logits
    loss = F.cross_entropy(logits, y)
    self.log_dict({'train_loss': loss})
    self.training_step_outputs.append({'loss': loss, 'logits': logits, 'y':y})
    return loss

  def on_train_epoch_end(self):
    # Concat batches
    outputs = self.training_step_outputs
    logits = torch.cat([x['logits'] for x in outputs])
    y = torch.cat([x['y'] for x in outputs])
    self.train_acc(logits, y)
    self.log_dict({
        'train_acc': self.train_acc,
      },
      on_step = False,
      on_epoch = True,
      prog_bar = True)
    self.training_step_outputs.clear()

  def validation_step(self, batch, batch_idx):
    x,y = batch
    logits = self.forward(x)
    loss = F.cross_entropy(logits, y)
    self.log_dict({'val_loss': loss})
    self.validation_step_outputs.append({'loss': loss, 'logits': logits, 'y':y})
    return loss

  def on_validation_epoch_end(self):
    # Concat batches
    outputs = self.validation_step_outputs
    logits = torch.cat([x['logits'] for x in outputs])
    y = torch.cat([x['y'] for x in outputs])
    self.val_acc(logits, y)
    self.log_dict({
        'val_acc': self.val_acc,
      },
      on_step = False,
      on_epoch = True,
      prog_bar = True)
    self.validation_step_outputs.clear()

  def test_step(self, batch, batch_idx):
    x,y = batch
    logits = self.forward(x)
    loss = F.cross_entropy(logits, y)
    self.log_dict({'test_loss': loss})
    self.test_step_outputs.append({'loss': loss, 'logits': logits, 'y':y})
    return loss

  def on_test_epoch_end(self):
    # Concat batches
    outputs = self.test_step_outputs
    logits = torch.cat([x['logits'] for x in outputs])
    y = torch.cat([x['y'] for x in outputs])
    self.test_acc(logits, y)
    self.log_dict({
        'test_acc': self.test_acc,
      },
      on_step = False,
      on_epoch = True,
      prog_bar = True)
    self.test_step_outputs.clear()

  def predict_step(self, batch):
        x, y = batch
        return self.model(x, y)

  def configure_optimizers(self):
    optimizer = optim.Adam(self.parameters(), lr=self.config.LEARNING_RATE)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode = 'min', patience = 3)
    lr_scheduler_dict = {
        "scheduler": lr_scheduler,
        "interval": "epoch",
         "monitor": "val_loss",
    }
    return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler_dict}

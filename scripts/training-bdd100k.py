import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
from helper import *

from torch.utils.data import DataLoader
from pathlib import Path
from torch.autograd import Variable
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from data_utils import LanesDataset, BDD100k
# from model import UNet
from models.Unet import UNet
from models.Dense_UNet import Dens_UNet
from loss import DiscriminativeLoss, CrossEntropyLoss2d
from torch.utils.tensorboard import SummaryWriter



train_img_dir = 'datasets/bdd100k/images/100k/train/'
train_label_dir = 'datasets/bdd100k/drivable_maps/labels/train/'
val_img_dir = 'datasets/bdd100k/images/100k/val/'
val_label_dir = 'datasets/bdd100k/drivable_maps/labels/val/'
model_dir = 'saved_models/'
exp_no = 2
logdir = 'runs/BDD100k_Experiment'+str(exp_no)
resize = (128,128)
SAMPLE_SIZE = 2000

# loading train data
train_dataset = BDD100k(train_img_dir,train_label_dir,resize=resize, transform=True, grayscale=False)
train_dataloader = DataLoader(train_dataset,batch_size=1, shuffle=False, pin_memory=True, num_workers=2)

# loading test data
test_dataset = BDD100k(val_img_dir,val_label_dir,resize=resize, transform=True)
test_dataloader = DataLoader(test_dataset,batch_size=1, shuffle=False, pin_memory=True, num_workers=2)

#loading model
model = Dens_UNet().cuda()

# Loss Function
criterion_bce_logits=torch.nn.BCEWithLogitsLoss()


# Optimizer
parameters = model.parameters()
optimizer = optim.SGD(parameters, lr=0.001, momentum=0.09, weight_decay=0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                 mode='min',
                                                 factor=0.1,
                                                 patience=10,
                                                 verbose=True)


# Train
model_dir = Path('saved_models/')
# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter(logdir)
global_step=0
running_loss = 0.0
best_loss = np.inf
for epoch in range(10):
    print(f'epoch : {epoch}')
    disc_losses = []
    bce_losses = []
    bce_ins_losses = []
    losses = []

    for i, batched in enumerate(tqdm(train_dataloader)):
        # with tqdm(total=len(train_dataloader.dataset), unit='img') as pbar:
        images, labels = batched
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()
        model.zero_grad()

        predict = model(images)
        loss = 0

        # BCE _ins_mask
        bce_loss = criterion_bce_logits(predict,labels)
        loss += bce_loss*3

        bce_ins_losses.append(bce_loss.cpu().data.numpy())
        losses.append(loss.cpu().data.numpy())

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:    # every 1000 mini-batches...

            # ...log the running loss
            writer.add_scalar('training loss',
                            running_loss / 1000,
                            epoch * len(train_dataloader) + i)

            # ...log a Matplotlib Figure showing the model's predictions on a
            # random mini-batch
            writer.add_figure('predictions vs. actuals',
                            plot_label_mask(model, images, labels, False),
                            global_step=epoch * len(train_dataloader) + i)
            running_loss = 0.0

        # break
    # disc_loss = np.mean(losses)
    bce_loss = np.mean(bce_losses)
    print(f'Total Loss: {loss:.4f}')
    print(f'BinaryCrossEntropyLoss: {bce_loss:.4f}')
    scheduler.step(bce_loss)
    if bce_loss < best_loss:
        best_loss = bce_loss
        print('Best Model!')
    modelname = 'model-DenseUnet-1.pth'
    torch.save(model.state_dict(), model_dir.joinpath(modelname))
    
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from pathlib import Path
from torch.autograd import Variable

from tqdm import tqdm

from data_utils import LanesDataset, BDD100k
from model import UNet
from loss import DiscriminativeLoss, CrossEntropyLoss2d


train_img_dir = 'datasets/bdd100k/images/100k/train/'
train_label_dir = 'datasets/bdd100k/drivable_maps/labels/train/'
resize = (128,128)
SAMPLE_SIZE = 2000
# loading data
train_dataset = BDD100k(train_img_dir,train_label_dir,resize=resize, transform=True)
train_dataloader = DataLoader(train_dataset,batch_size=1, shuffle=False, pin_memory=True, num_workers=2)

#loading model
model = UNet().cuda()

# Loss Function
criterion_disc = DiscriminativeLoss(delta_var=0.5,
                                    delta_dist=1.5,
                                    norm=2,
                                    usegpu=True).cuda()
criterion_ce = CrossEntropyLoss2d()

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

best_loss = np.inf
for epoch in range(10):
    print(f'epoch : {epoch}')
    disc_losses = []
    bce_losses = []
    bce_ins_losses = []
    losses = []

    for i, batched in enumerate(tqdm(train_dataloader)):
        # with tqdm(total=len(train_dataloader.dataset), unit='img') as pbar:
        images, ins_labels = batched
        images = Variable(images).cuda()
        # sem_labels = Variable(sem_labels).cuda()
        ins_labels = Variable(ins_labels).cuda()
        model.zero_grad()

        sem_predict, ins_predict = model(images.float())
        loss = 0

        # # Discriminative Loss
        # disc_loss = criterion_disc(ins_predict,
        #                         ins_labels,
        #                         [2] * len(images))
        # loss += disc_loss*0.5
        # if i%50 == 0:
        #     print(f'Disc Loss: {disc_loss.cpu().data.numpy()}')
        # disc_losses.append(disc_loss.cpu().data.numpy())

        # BCE _bin_mask
        # bce_loss=criterion_bce_logits(sem_predict,sem_labels)
        # loss += bce_loss
        # if i%50 == 0:
        #     print(f'BCE Loss: {bce_loss.cpu().data.numpy()}')
        # bce_losses.append(bce_loss.cpu().data.numpy())

        # BCE _ins_mask
        bce_loss_ins=criterion_bce_logits(ins_predict,ins_labels)
        loss += bce_loss_ins*3
        if i%50 == 0:
            print(f'BCE Loss Ins: {bce_loss_ins.cpu().data.numpy()}')
        bce_ins_losses.append(bce_loss_ins.cpu().data.numpy())
        # Cross Entropy Loss
        # _, sem_labels_ce = sem_labels.max(1)
        # ce_loss = criterion_ce(ins_predict,
        #                            ins_labels.long())
        # loss += ce_loss
        # ce_losses.append(ce_loss.cpu().data.numpy()[0])

        losses.append(loss.cpu().data.numpy())

        loss.backward()
        optimizer.step()

        # break
    # disc_loss = np.mean(losses)
    bce_loss = np.mean(bce_losses)
    print(f'Total Loss: {loss:.4f}')
    print(f'BinaryCrossEntropyLoss: {bce_loss:.4f}')
    scheduler.step(bce_loss)
    if bce_loss < best_loss:
        best_loss = bce_loss
        print('Best Model!')
    modelname = 'model-1.pth'
    torch.save(model.state_dict(), model_dir.joinpath(modelname))
        
    # break

# for img, bmask, imask in train_dataloader:

#     print(img.shape)
#     print(bmask.shape)
#     print(imask.shape)
#     sem_predict, ins_predict = model(img.cuda())
#     print(ins_predict.shape)
#     disc_loss = criterion_disc(ins_predict.float(),
#                                    imask.cuda().float(),
#                                    [4] * len(img.cuda()))
#     print(disc_loss.cpu().data.numpy())

#     break

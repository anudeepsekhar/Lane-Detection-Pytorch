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

from data_utils import LanesDataset
from model import UNet
from loss import DiscriminativeLoss, CrossEntropyLoss2d


root_dir = '/home/anudeep/lane-detection/dataset/'

df = pd.read_csv(os.path.join(root_dir,'data/paths.csv'))

X_train, X_test, y_train, y_test = train_test_split(
    df.img_paths, df.mask_paths, test_size=0.2, random_state=42)

# loading data
train_dataset = LanesDataset(np.array(X_train[0:5000]), np.array(y_train[0:5000]), resize=(128,128))
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
model_dir = Path(root_dir+'models/')

best_loss = np.inf
for epoch in range(1):
    print(f'epoch : {epoch}')
    disc_losses = []
    bce_losses = []

    for i, batched in enumerate(tqdm(train_dataloader)):
        # with tqdm(total=len(train_dataloader.dataset), unit='img') as pbar:
        images, sem_labels, ins_labels = batched
        images = Variable(images).cuda()
        sem_labels = Variable(sem_labels).cuda()
        ins_labels = Variable(ins_labels).cuda()
        model.zero_grad()

        sem_predict, ins_predict = model(images.float())
        loss = 0

        # Discriminative Loss
        disc_loss = criterion_disc(ins_predict,
                                ins_labels,
                                [2] * len(images))
        loss += disc_loss
        if i%50 == 0:
            print(f'Disc Loss: {disc_loss.cpu().data.numpy()}')
        disc_losses.append(disc_loss.cpu().data.numpy())

        # BCE 
        bce_loss=criterion_bce_logits(sem_predict,sem_labels)
        loss += bce_loss*3
        if i%50 == 0:
            print(f'BCE Loss: {bce_loss.cpu().data.numpy()}')
        bce_losses.append(bce_loss.cpu().data.numpy())
        # Cross Entropy Loss
        # _, sem_labels_ce = sem_labels.max(1)
        # ce_loss = criterion_ce(ins_predict,
        #                            ins_labels.long())
        # loss += ce_loss
        # ce_losses.append(ce_loss.cpu().data.numpy()[0])

        loss.backward()
        optimizer.step()
        # break
    disc_loss = np.mean(disc_losses)
    bce_loss = np.mean(bce_losses)
    print(f'DiscriminativeLoss: {disc_loss:.4f}')
    print(f'BinaryCrossEntropyLoss: {bce_loss:.4f}')
    scheduler.step(disc_loss)
    if disc_loss < best_loss:
        best_loss = disc_loss
        print('Best Model!')
        modelname = 'model-2.pth'
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

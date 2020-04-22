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
from torch.utils.tensorboard import SummaryWriter


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
for epoch in range(10):
    print(f'epoch : {epoch}')
    disc_losses = []
    bce_losses = []
    bce_ins_losses = []

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
        loss += disc_loss*0.5
        if i%50 == 0:
            print(f'Disc Loss: {disc_loss.cpu().data.numpy()}')
        disc_losses.append(disc_loss.cpu().data.numpy())

        # BCE _bin_mask
        bce_loss=criterion_bce_logits(sem_predict,sem_labels)
        loss += bce_loss*3
        if i%50 == 0:
            print(f'BCE Loss: {bce_loss.cpu().data.numpy()}')
        bce_losses.append(bce_loss.cpu().data.numpy())

        # BCE _ins_mask
        bce_loss_ins=criterion_bce_logits(ins_predict,ins_labels)
        loss += bce_loss_ins
        if i%50 == 0:
            print(f'BCE Loss Ins: {bce_loss_ins.cpu().data.numpy()}')
        bce_ins_losses.append(bce_loss_ins.cpu().data.numpy())

        loss.backward()
        optimizer.step()
        # break
        running_loss += loss.item()
        if i % 100 == 99:    # every 1000 mini-batches...

            # ...log the running loss
            writer.add_scalar('training loss',
                            running_loss / 1000,
                            epoch * len(train_dataloader) + i)

            # ...log a Matplotlib Figure showing the model's predictions on a
            # random mini-batch
            writer.add_graph(model, images)
            writer.add_figure('predictions vs. actuals',
                            plot_label_mask(model, images, labels, False),
                            global_step=epoch * len(train_dataloader) + i)
            running_loss = 0.0
    disc_loss = np.mean(disc_losses)
    bce_loss = np.mean(bce_losses)
    print(f'DiscriminativeLoss: {disc_loss:.4f}')
    print(f'BinaryCrossEntropyLoss: {bce_loss:.4f}')
    scheduler.step(bce_loss)
    if bce_loss < best_loss:
        best_loss = bce_loss
        print('Best Model!')
        modelname = 'model-3.pth'
        torch.save(model.state_dict(), model_dir.joinpath(modelname))

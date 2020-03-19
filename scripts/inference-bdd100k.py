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
import matplotlib.pyplot as plt

from data_utils import LanesDataset, BDD100k
from model import UNet
from loss import DiscriminativeLoss, CrossEntropyLoss2d

from utils import gen_color_img


resize = (128,128)
val_img_dir = 'datasets/bdd100k/images/100k/val/'
val_label_dir = 'datasets/bdd100k/drivable_maps/labels/val/'

# loading data
test_dataset = BDD100k(val_img_dir,val_label_dir,resize=resize, transform=True)
test_dataloader = DataLoader(test_dataset,batch_size=1, shuffle=False, pin_memory=True, num_workers=2)

#loading model
model = UNet().cuda()
model.eval()
root_dir = '/'
model_dir = Path('saved_models/')
model_path = model_dir.joinpath('model-1.pth')

param = torch.load(model_path)
model.load_state_dict(param)

# Inference
images = []
sem_pred = []
ins_pred = []
sem_labels = []
ins_labels = []
for i, batched in enumerate(test_dataloader):
    if i<10:
        images_, ins_labels_ = batched
        images.append(images_.numpy())
        # sem_labels.append(sem_labels_.numpy())
        ins_labels.append(ins_labels_.numpy())
        images_ = Variable(images_, volatile=True).cuda()
        ins_pred_,hidden = model(images_)
        print(ins_pred_.shape)
        # print(ins_pred_.cpu())
        # sem_pred.append(sem_pred_.cpu().data.numpy())
        ins_pred.append(ins_pred_.cpu().data.numpy())
    else:
        break

images = np.concatenate(images)[:, 0].astype(np.uint8)
# sem_pred = np.concatenate(sem_pred)[:,0]
ins_pred = np.concatenate(ins_pred)
print(len(ins_pred))
# sem_labels = np.concatenate(sem_labels)[:,0]
ins_labels = np.concatenate(ins_labels)

# fig, axes = plt.subplots(3, 7, figsize=(15, 15))
# plt.gray()

fig2, axes2 = plt.subplots(10, 3, figsize=(10, 30))

# for i, ax_ in enumerate(axes):
#     # color_img = gen_color_img(sem_pred[i], ins_pred[i], 4)
for i, ax in enumerate(axes2):
    ax[0].imshow(images[i])
    # axes2[0][1].imshow(sem_pred[0])
    # print(sem_pred[i].shape)
    # for k in range(1):
    ax[1].imshow(1 - ins_pred[i][0])

    # ax[0].imshow(images[0])
    # axes2[1][1].imshow(sem_labels[0])
    # print(sem_pred[i].shape)
    # for k in range(1):
    ax[2].imshow(ins_labels[i][0])

plt.show()
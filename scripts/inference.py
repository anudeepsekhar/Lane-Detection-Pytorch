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

from data_utils import LanesDataset
from model import UNet
from loss import DiscriminativeLoss, CrossEntropyLoss2d

from utils import gen_color_img



root_dir = '/home/anudeep/lane-detection/dataset/'

df = pd.read_csv(os.path.join(root_dir,'data/paths.csv'))

X_train, X_test, y_train, y_test = train_test_split(
    df.img_paths, df.mask_paths, test_size=0.2, random_state=42)

# loading data
test_dataset = LanesDataset(np.array(X_test[0:3]), np.array(y_test[0:3]), resize=(128,128))
test_dataloader = DataLoader(test_dataset,batch_size=1, shuffle=False, pin_memory=True, num_workers=2)

#loading model
model = UNet().cuda()
model.eval()

model_dir = Path(root_dir+'models/')
model_path = model_dir.joinpath('model-3.pth')

param = torch.load(model_path)
model.load_state_dict(param)

# Inference
images = []
sem_pred = []
ins_pred = []
for batched in test_dataloader:
    images_, sem_labels, ins_labels = batched
    images.append(images_.numpy())
    images_ = Variable(images_, volatile=True).cuda()
    sem_pred_, ins_pred_ = model(images_)
    sem_pred.append(sem_pred_.cpu().data.numpy())
    ins_pred.append(ins_pred_.cpu().data.numpy())

images = np.concatenate(images)[:, 0].astype(np.uint8)
sem_pred = np.concatenate(sem_pred)[:,0]
ins_pred = np.concatenate(ins_pred)

fig, axes = plt.subplots(3, 7, figsize=(15, 15))
plt.gray()

for i, ax_ in enumerate(axes):
    # color_img = gen_color_img(sem_pred[i], ins_pred[i], 4)
    ax_[0].imshow(images[i])
    ax_[1].imshow(sem_pred[i])
    print(sem_pred[i].shape)
    for k in range(5):
        ax_[2+k].imshow(ins_pred[i][k])

plt.show()
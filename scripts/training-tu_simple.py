import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
from helper import *
 
from torch.utils.data import DataLoader, Subset
from pathlib import Path
from torch.autograd import Variable
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from data_utils import LanesDataset, BDD100k, TUDataset, TU_Lane_Dataset
from loss import DiceLoss, BinaryFocalLoss, FocalLoss_Ori
# from model import UNet
from models.Unet import UNet
# from models.Dense_UNet import Dens_UNet
# from models.Dense_UNet3 import Dense_UNet2
# from models.covidNet import Net3
# from models.drivenet import driveNet
# from models.UnetResnet import UNetResnet
from models.HGNet import HGNet
from loss import DiscriminativeLoss, CrossEntropyLoss2d
from torch.utils.tensorboard import SummaryWriter



train_img_dir = 'datasets/bdd100k/images/100k/train/'
train_label_dir = 'datasets/bdd100k/drivable_maps/labels/train/'
val_img_dir = 'datasets/bdd100k/images/100k/val/'
val_label_dir = 'datasets/bdd100k/drivable_maps/labels/val/'
model_dir = 'saved_models/'
exp_name = 'HGNet-2'
logdir = 'runs-HGNet/TU_Experiment'+exp_name
SAMPLE_SIZE = 15000

def getLaneDataset():
    resize = (256, 256)
    labels_json = os.path.join( '/home/anudeep/lane-detection/dataset', 'label_data.json')
    data = pd.read_json(labels_json, lines=True)
    image_files = []
    mask_files = []
    for n, file_path in enumerate(data['raw_file'].to_numpy()):
        path_list = file_path.split('/')[1:-1]
        image_files += [ os.path.join('/home/anudeep/lane-detection/dataset', 'clips', *path_list, str(i)+'.jpg') for i in range(1, 21)]
        mask_files += [ os.path.join( '/home/anudeep/lane-detection/dataset', 'masks', *path_list, str(i)+'.tiff') for i in range(1,21)]
        # mask2_files += [ os.path.join( '/home/anudeep/lane-detection/dataset', 'lane_masks', *path_list, str(i)+'.tiff') for i in range(1,21)]
    image_files = np.array(image_files)
    mask_files = np.array(mask_files)
    # mask2_files = np.array(mask2_files)
    
    X_train, X_val, y_train, y_val = train_test_split( 
        image_files, mask_files, test_size=0.3)

    return ( 
        TU_Lane_Dataset(X_train, y_train, resize=resize, transform=True),
        TU_Lane_Dataset(X_val, y_val, resize=resize, transform=False)
        )

train_dataset, test_dataset = getLaneDataset()

# loading train data
# train_dataset = BDD100k(train_img_dir,train_label_dir,resize=resize, transform=True, grayscale=False)
train_subset = Subset(train_dataset, range(0,15000))
# train_dataloader = DataLoader(train_dataset,batch_size=1, shuffle=False, pin_memory=True, num_workers=2)

# df = pd.read_csv('datasets/tu_simple/data/lane_mask.csv')

# X_train, X_test, y_train, y_test = train_test_split(
    # df.img_paths, df.mask_paths, test_size=0.2, random_state=42)

# # loading data
# train_dataset = TUDataset(np.array(X_train[0:SAMPLE_SIZE]), np.array(y_train[0:SAMPLE_SIZE]), resize=(96,96))
train_dataloader = DataLoader(train_subset,batch_size=1, shuffle=False, pin_memory=True, num_workers=2)

# # loading test data
# test_dataset = TUDataset(np.array(X_train[0:SAMPLE_SIZE]), np.array(y_train[0:SAMPLE_SIZE]), resize=(96,96))
test_dataloader = DataLoader(test_dataset,batch_size=1, shuffle=False, pin_memory=True, num_workers=2)

#loading model
# model = UNetResnet(n_class=1).cuda()
# model = driveNet(n_classes=1).cuda()
# model = torch.load('/home/anudeep/repos/Lane-Detection-Pytorch/scripts/saved_models/model-UnetResnet-1.pth')
# for param in model.block3.parameters():
#     param.requires_grad = True
# for param in model.block2.parameters():
#     param.requires_grad = True
# for param in model.block1.parameters():
#     param.requires_grad = True
# model = torch.load('/home/anudeep/repos/Lane-Detection-Pytorch/scripts/saved_models/model-8.pth').cuda()
# model = UNet(n_classes=2).cuda()
model = HGNet().cuda()
# model = torch.load('/home/anudeep/repos/Lane-Detection-Pytorch/scripts/saved_models/model-UNET-4.pth')


# Loss Function
criterion_bce_logits=torch.nn.BCEWithLogitsLoss()
criterion_bce=torch.nn.BCELoss()
criterion_mse = nn.MSELoss()
criterion_disc = DiceLoss()
criterion_focal = BinaryFocalLoss()



# Optimizer
parameters = model.parameters()
optimizer = optim.SGD(parameters, lr=0.001, momentum=0.09, weight_decay=0.0001)
# optimizer = optim.Adam(parameters, lr=0.001,weight_decay=0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                 mode='min',
                                                 factor=0.01,
                                                 patience=10,
                                                 verbose=True)


# Train
model_dir = Path('saved_models/')
# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter(logdir)
global_step=0
running_loss = 0.0
best_loss = np.inf
eps = 10**-10
for epoch in range(20):
    print(f'epoch : {epoch}')
    disc_losses = []
    bce_losses = []
    bce_ins_losses = []
    losses = []

    for i, batched in enumerate(tqdm(train_dataloader)):
        # with tqdm(total=len(train_dataloader.dataset), unit='img') as pbar:
        images, labels, points = batched
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()
        points = Variable(points).cuda()
        model.zero_grad()
        result1, pred_pts, predict = model(images)
        loss = 0

        # BCE _ins_mask
        predict = F.sigmoid(predict)
        bce_loss = criterion_bce(predict,labels)
        loss += 3*bce_loss

        mse_loss = criterion_mse(points, pred_pts)
        loss+=mse_loss

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
            writer.add_graph(model, images)
            writer.add_figure('predictions vs. actuals',
                            plot_tu_data(images, labels, predict),
                            global_step=epoch * len(train_dataloader) + i)
            running_loss = 0.0

    scheduler.step(bce_loss)
    if bce_loss < best_loss:
        best_loss = bce_loss
        print('Best Model!')
    modelname = 'model-'+ exp_name +'.pth'
    torch.save(model, model_dir.joinpath(modelname))
    
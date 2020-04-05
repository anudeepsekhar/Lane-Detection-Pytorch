import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data_utils import LanesDataset, BDD100k
from torch.utils.data import DataLoader

from models.covidNet import Net

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

img, mask = iter(train_dataloader).next()
print(img.shape)
print(mask.shape)

model = Net()

pred = model(img)



# print(model)
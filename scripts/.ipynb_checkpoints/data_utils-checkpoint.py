import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import cv2
import numpy as np
from glob import glob
class LanesDataset(torch.utils.data.Dataset):
    """Some Information about LanesDataset"""
    def __init__(self, img_paths, mask_paths, train= True, resize=None, channel_first=True, transform=True):
        super(LanesDataset, self).__init__()
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.train = train
        self.resize = resize
        self.transform = transform
        self.channel_first = channel_first

    def _split_instance_mask(self, label_instance_img):
        no_of_instances = 5
        ins = np.zeros((no_of_instances, label_instance_img.shape[0],label_instance_img.shape[1]))
        for _ch, label in enumerate([i for i in np.unique(label_instance_img)[1:] if i>0]):
            ins[_ch, label_instance_img == label] = 1
        return ins

    def _get_binary_mask(self, label_img):
        label_binary = np.zeros([label_img.shape[0], label_img.shape[1]], dtype=np.uint8)
        mask = np.nonzero(label_img)
        label_binary[mask] = 1
        return np.expand_dims(label_binary, axis=0)


    def __getitem__(self, index):
        img = cv2.imread(self.img_paths[index])
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        if self.resize is not None:
            img = cv2.resize(img, self.resize, cv2.INTER_NEAREST)

        if self.channel_first:
            img = np.transpose(img, (2, 0, 1))
        
        mask = cv2.imread(self.mask_paths[index], cv2.IMREAD_UNCHANGED)

        if self.resize is not None:
            instance_mask = self._split_instance_mask(mask)
            instance_masks = []
            for i in range(len(instance_mask)):
                instance_masks.append(cv2.resize(instance_mask[i], self.resize, cv2.INTER_NEAREST))
            mask = cv2.resize(mask, self.resize, cv2.INTER_NEAREST)

        
        binary_mask = self._get_binary_mask(mask)
        # binary_mask = cv2.resize(binary_mask, self.resize, cv2.INTER_NEAREST)



        if self.train:
            img = torch.tensor(img).float()
            binary_mask = torch.tensor(binary_mask).float()
            instance_mask = torch.tensor(instance_mask).float()


        return img, binary_mask, np.array(instance_masks)

    def __len__(self):
        return len(self.img_paths)



class TUDataset(torch.utils.data.Dataset):
    """Some Information about LanesDataset"""
    def __init__(self, img_paths, mask_paths, train= True, resize=None, channel_first=True, transform=True):
        super(TUDataset, self).__init__()
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.train = train
        self.resize = resize
        self.transform = transform
        self.channel_first = channel_first

    def __getitem__(self, index):
        img = cv2.imread(self.img_paths[index])
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = img/255
        mask = cv2.imread(self.mask_paths[index], cv2.IMREAD_UNCHANGED)
        mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
        mask = mask/255
        if self.resize is not None:
            img = cv2.resize(img, self.resize, cv2.INTER_NEAREST)
            mask = cv2.resize(mask, self.resize, cv2.INTER_NEAREST)

        if self.channel_first:
            img = np.transpose(img, (2, 0, 1))
            # mask = np.transpose(mask, (2, 0, 1))
            mask = np.expand_dims(mask, axis=0)
        
        if self.train:
            img = torch.tensor(img).float()
            mask = torch.tensor(mask).float()
        return img, mask

    def __len__(self):
        return len(self.img_paths)


class BDD100k(torch.utils.data.Dataset):
    """Some Information about BDD100k"""
    def __init__(self, img_dir, masks_dir, resize=None, channels_first=True, transform=True,train= True, grayscale=False):
        self.train = train
        self.img_dir = img_dir
        self.masks_dir = masks_dir
        self.img_paths = glob(self.img_dir+'*jpg')
        self.label_paths = glob(self.masks_dir+'*png')
        # print(self.label_paths)
        self.channels_first = channels_first
        self.resize = resize
        self.grayscale = grayscale
        self.transform = transform

    def __len__(self):
        return len(self.label_paths)

    def _split_instance_mask(self, label_instance_img):
        no_of_instances = 2
        ins = np.zeros((no_of_instances, label_instance_img.shape[0],label_instance_img.shape[1]))
        for _ch, label in enumerate([i for i in np.unique(label_instance_img)[1:] if i>0]):
            ins[_ch, label_instance_img == label] = 1
        return ins

    def __getitem__(self, index):
        label_path = self.label_paths[index]
        img_id = label_path.split('.')[0].split('/')[-1].split('_')[0]
        img_path = self.img_dir+img_id+'.jpg'
        img = cv2.imread(img_path)
        if self.grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # print('shape:', img.shape)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


        if not self.resize is None:
            img = cv2.resize(img, self.resize, interpolation=cv2.INTER_CUBIC)
        if self.channels_first:
            if self.grayscale:
                img = np.expand_dims(img, axis=0)
            else:
                img = np.transpose(img, (2, 0, 1))

        mask = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
        if self.resize is not None:
            instance_mask = self._split_instance_mask(mask)
            instance_masks = []
            for i in range(len(instance_mask)):
                instance_masks.append(cv2.resize(instance_mask[i], self.resize, cv2.INTER_NEAREST))
            mask = cv2.resize(mask, self.resize, cv2.INTER_NEAREST)
        # print(mask.shape)

        if self.train:
            img = torch.tensor(img).float()
            # binary_mask = torch.tensor(binary_mask).float()
            instance_mask = torch.tensor(instance_mask).float()
        return img, np.array(instance_masks)

class TU_Lane_Dataset(torch.utils.data.Dataset):
    def __init__(self, images, masks, resize=None, channels_first=True, transform=True):
        self.X = images
        self.y = masks
        self.channels_first = channels_first
        self.resize = resize
        self.transform = transform

    def __len__(self):
        return self.y.shape[0]

    def _get_binary_mask(self, label_img):
        label_binary = np.zeros([label_img.shape[0], label_img.shape[1]], dtype=np.uint8)
        mask = np.nonzero(label_img)
        label_binary[mask] = 1
        return np.expand_dims(label_binary, axis=0)

    def __getitem__(self, index):
        img = cv2.imread(self.X[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if not self.resize is None:
            img = cv2.resize(img, self.resize, interpolation=cv2.INTER_CUBIC)

        img = img/255

        if self.channels_first:
            img = np.transpose(img, (2, 0, 1))

        mask = cv2.imread(self.y[index], cv2.IMREAD_GRAYSCALE)
        if not self.resize is None:
            mask = cv2.resize(mask, self.resize, interpolation=cv2.INTER_NEAREST)
        mask = mask.astype(np.uint8)

        mask[mask==3] = 0
        mask[mask==4] = 0
        x1,y1 = np.where(mask==1)
        x2,y2 = np.where(mask==2)
        offset_x = 3
        offset_y = 15

        try:
            points = np.array([
                x1[0]-offset_x, y1[0]-offset_y,
                x1[-1]+offset_x, y1[-1]-offset_y, 
                x2[-1]+offset_x, y2[-1]+offset_y, 
                x2[0]-offset_x, y2[0]+offset_y], dtype=np.int32
                )
        except:
            print(self.y[index])
            print(x1, y1, x2, y2)
            return [], [], []

        points[ points>= mask.shape[0] ] = mask.shape[0] - 1
        mask[mask==2] = 1
        mask = mask[np.newaxis, :, :]
        img = torch.tensor(img).float()
        mask = torch.tensor(mask).float()
        return img, mask #, points
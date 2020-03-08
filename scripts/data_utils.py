import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import cv2
import numpy as np

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
        for _ch, label in enumerate(np.unique(label_instance_img)[1:]):
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
            img = cv2.resize(img, self.resize, cv2.INTER_CUBIC)

        if self.channel_first:
            img = np.transpose(img, (2, 0, 1))
        
        mask = cv2.imread(self.mask_paths[index], cv2.IMREAD_UNCHANGED)

        if self.resize is not None:
            mask = cv2.resize(mask, self.resize, cv2.INTER_LINEAR)

        instance_mask = self._split_instance_mask(mask)
        binary_mask = self._get_binary_mask(mask)


        if self.train:
            img = torch.tensor(img).float()
            binary_mask = torch.tensor(binary_mask).float()
            instance_mask = torch.tensor(instance_mask).float()


        return img, binary_mask, instance_mask

    def __len__(self):
        return len(self.img_paths)
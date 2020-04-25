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
        # self.label_paths = lane_masks
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

    def points_interpolation(self, x1, y1, num_points=15):
        x2 = np.unique(x1)
        y2 = []
        for i in x2: 
            tx = np.where(x1==i)[0] 
            ty = np.floor(np.min([tx[0],tx[-1]]) + np.absolute((tx[0]-tx[-1])/2)) 
            y2.append( y1[int(ty)] )

        xvals = np.linspace(x2[0], x2[-1], num_points)
        yvals = np.interp(xvals, x2, y2)
        return xvals, yvals

    def get_x(self, y, mask, l):
        xi = np.where(mask[y,:]==l)
        xi = xi[0]
        xx = (xi[0]+xi[-1])/2
        return xx

    def get_pts_from_lane(self, idx, mask):
        x,y = np.where(mask==idx)
        y_vals = np.linspace(min(x),max(x),15)
        y_vals = np.array(y_vals).astype(np.uint16)
        x_vals = [self.get_x(y, mask, idx) for y in y_vals]
        return x_vals, y_vals

    def get_pts(self, mask):
        x1, y1 = self.get_pts_from_lane(1,mask)
        x2, y2 = self.get_pts_from_lane(2,mask)
        pts = np.array([x1,y1,x2,y2])
        return pts.reshape(-1)

    def __getitem__(self, index):
        img = cv2.imread(self.X[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if not self.resize is None:
            img = cv2.resize(img, self.resize, interpolation=cv2.INTER_CUBIC)

        img = img/255

        if self.channels_first:
            img = np.transpose(img, (2, 0, 1))
            

        mask = cv2.imread(self.y[index], cv2.IMREAD_GRAYSCALE)
        # mask2 = cv2.imread(self.y[index].replace('masks', 'lane_masks'), cv2.IMREAD_GRAYSCALE)
        if not self.resize is None:
            mask = cv2.resize(mask, self.resize, interpolation=cv2.INTER_NEAREST)
            # mask2 = cv2.resize(mask2, self.resize, interpolation=cv2.INTER_NEAREST)

        mask = mask.astype(np.uint8)
        # mask2 = mask2.astype(np.uint8)
        # mask2 = np.transpose(img, (2, 0, 1))
        # mask2 = np.expand_dims(mask2, axis=0)

        mask[mask==3] = 0
        mask[mask==4] = 0
        # points = self.get_pts(mask)
        x1,y1 = np.where(mask==1)
        x2,y2 = np.where(mask==2)

        l1_x, l1_y = self.points_interpolation(x1, y1)
        l2_x, l2_y = self.points_interpolation(x2, y2)

        pts1 = [[i,j] for i, j in zip(l1_x, l1_y)] 
        pts2 = [[i,j] for i, j in zip(l2_x, l2_y)]
        pts = np.array(pts1 + pts2)
        points = pts.reshape(-1)
        # # print(points)
        mask[mask==2] = 1
        # masks = []
        # masks.append(mask)
        # masks.append(mask2/255)
        # cv2.imwrite('/home/anudeep/temp/mask2.jpg',mask2)
        mask = mask[np.newaxis, :, :]

        img = torch.tensor(img).float()
        masks = torch.tensor(mask).float()
        points = torch.tensor(points).float()
        # mask2 = torch.tensor(mask2).float()
        return img, masks, points

'''
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
        '''
        
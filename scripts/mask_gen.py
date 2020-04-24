import os
import cv2
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

def imshow(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    
def get_lane_mask(sample,lane_idx):
    points_lane = []
    h_max = np.max(data['h_samples'][sample])
    h_min = np.min(data['h_samples'][sample])
    x_idx = data['lanes'][sample][lane_idx]
    y_idx = data['h_samples'][sample]
    for x,y in zip(x_idx,y_idx):
        offset = (y-h_min)/5
    #     print(offset)
        if x>-100:
            points_lane.append([x-offset/2,y])
    x_idx_=x_idx.copy()
    y_idx_=y_idx.copy()
    x_idx_.reverse()
    y_idx_.reverse()
    for x,y in zip(x_idx_,y_idx_):
        offset = (y-h_min)/5
    #     print(offset)
        if x>-100:
            points_lane.append([x+offset/2,y])
    return points_lane

def create_lane_mask(img_raw,sample):
    colors = [[255,0,0],[0,255,0],[0,0,255],[0,255,255]]
    laneMask = np.zeros(img_raw.shape, dtype=np.uint8)
    for lane_idx in range(len(data.lanes[sample])):
        points_lane = get_lane_mask(sample,lane_idx)
        if len(points_lane)>0: 
            pts = np.array(points_lane, np.int32)
            pts = pts.reshape((-1,1,2))
            laneMask = cv2.fillPoly(laneMask,[pts],colors[lane_idx])
            colors = [[255,0,0],[0,255,0],[0,0,255],[0,255,255]]
            # create grey-scale label image
            label = np.zeros((720,1280),dtype = np.uint8)
            for i in range(len(colors)):
                label[np.where((laneMask == colors[i]).all(axis = 2))] = i+1
        else: continue
    return(img_raw, label)


data_dir = '/home/anudeep/lane-detection/dataset'
data = pd.read_json(os.path.join(data_dir, 'label_data.json'), lines=True)

for i in tqdm(range(0,len(data.raw_file))):
    img_path = data.raw_file[i]
    img_path = os.path.join(data_dir,img_path)
#     print('Reading from: ', img_path)
    path_list = img_path.split('/')[:-1]
    mask_path_dir = os.path.join(*path_list)

    img_raw = cv2.imread(img_path)
    img_, mask = create_lane_mask(img_raw,i)

#     fig = plt.figure(figsize=(15,20))
#     plt.subplot(211)
#     imshow(img_raw)
#     plt.subplot(212)
#     print(mask.shape)
#     plt.imshow(mask)

    mask_path_dir = mask_path_dir.replace('clips', 'masks')
#     print('Saving to: ', mask_path_dir)
    try:
        os.makedirs(mask_path_dir)
    except:
        pass

    for i in range(1, 21):
#         print('/'+os.path.join( mask_path_dir, f'{i}.tiff'))
        cv2.imwrite('/'+os.path.join( mask_path_dir, f'{i}.tiff'), mask)

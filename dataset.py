"""
Preparing datasets and labeling with csv file

@author: Shen Jie Koh
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import os
import pandas as pd

def video_to_tensor(vid):
    '''
    vid (numpy.ndarray) : Video to be converted to tensor.
    Convert video numpy array to torch.FloatTensor of shape (C x T x H x W)
    '''
    return torch.from_numpy(vid.transpose([3,0,1,2]))

def load_rgb_frames(numpy):
    return np.asarray(np.load(numpy)[:, :, :, [2,1,0]], dtype=np.float32)

def load_grayscale_frames(numpy):
    binary = np.load(numpy)
    binary = np.expand_dims(binary, axis = 3)
    return np.asarray(binary, dtype=np.float32)

class FloorDataset(Dataset):
    def __init__(self, csv_file, root_dir):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        vid_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        numpy_path = vid_path.replace('mp4', 'npy')
        if 'rgb' in numpy_path:
            vid = load_rgb_frames(numpy_path)
        else:
            vid = load_grayscale_frames(numpy_path)
        label = torch.zeros((14,11))
        y_label = self.annotations.iloc[index, 1]
        
        if y_label == 'B1':
            label[13,:] = 1
        elif y_label == 'B2':
            label[0,:] = 1
        else:
            label[int(y_label),:] = 1
        video = video_to_tensor(vid)
            
        return video, label, vid_path
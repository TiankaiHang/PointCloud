from __future__ import print_function, division

import os
import numpy as np
import open3d as o3d

import torch
import torch.utils.data as data

from utils import *

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

folder2category = {'02691156':   'Airplane'	,
                   '02773838':        'Bag'	,
                   '02954340':        'Cap'	,
                   '02958343':        'Car'	,
                   '03001627':      'Chair'	,
                   '03261776':   'Earphone'	,
                   '03467517':     'Guitar'	,
                   '03624134':      'Knife'	,
                   '03636649':       'Lamp'	,
                   '03642806':     'Laptop'	,
                   '03790512':  'Motorbike'	,
                   '03797390':        'Mug'	,
                   '03948459':     'Pistol'	,
                   '04099429':     'Rocket'	,
                   '04225987': 'skateboard'	,
                   '04379243':      'Table'	}

folder2class =    { '02691156'	:  0,
                    '02773838'	:  1,
                    '02954340'	:  2,
                    '02958343'	:  3,
                    '03001627'	:  4,
                    '03261776'	:  5,
                    '03467517'	:  6,
                    '03624134'	:  7,
                    '03636649'	:  8,
                    '03642806'	:  9,
                    '03790512'	: 10,
                    '03797390'	: 11,
                    '03948459'	: 12,
                    '04099429'	: 13,
                    '04225987'	: 14,
                    '04379243'	: 15}

class shapeNetDataset(data.Dataset):
    def __init__(self,
                 root=BASE_DIR,
                 num_points=2500,
                 train=True,
                 classification=True,
                 class_choice=None,
                 data_augmentation=True):
        self.root = root
        self.num_points = num_points
        self.classification = classification
        self.train = train
        self.class_choice = class_choice
        self.data_augmentation = data_augmentation
        
        self.datapath = []
        self.labelpath = []

        datasetFolder = os.path.join(BASE_DIR,'dataset')

        if self.train:
            dataFolder = os.path.join(datasetFolder, 'train_data')
            labelFolder = os.path.join(datasetFolder, 'train_label')
        else:
            # Haven't got the test label
            dataFolder = os.path.join(datasetFolder, 'val_data')
            labelFolder = os.path.join(datasetFolder, 'val_label')

        for folder in os.listdir(dataFolder):
            pointdatafold = os.path.join(dataFolder, folder)
            pointlabelfold = os.path.join(labelFolder, folder)
            for i, j in zip(os.listdir(pointdatafold), os.listdir(pointlabelfold)):
                self.datapath.append(os.path.join(pointdatafold, i))
                self.labelpath.append(os.path.join(pointlabelfold, j))


    def __getitem__(self, index):
        filename_data = self.datapath[index]
        filename_label = self.labelpath[index]

        point_data = o3d.io.read_point_cloud(filename_data, format='xyz')
        point_data = np.asarray(point_data.points, dtype=np.float32)
        sample_index = np.random.choice(len(point_data), self.num_points, replace=True)
        point_data = point_data[sample_index, :]
        point_data = point_data - point_data.mean(axis=0)
        sigma = np.max(np.sum(point_data * point_data, axis=1), axis=0)
        point_data = point_data / sigma

        point_class = folder2class[filename_data.split('\\')[-2]]
        point_label = seg2array(filename_label)

        # Add some noise to do augmentation
        if self.data_augmentation:
            point_data += np.random.normal(0, 0.02, size=point_data.shape)


        point_data = torch.from_numpy(point_data)
        point_class = torch.from_numpy(np.array([point_class]).astype(np.int64))
        point_label = torch.from_numpy(point_label).view(-1)[sample_index]

        # print(len(point_data))

        if self.classification:
            return point_data, class2onehot(point_class)
        else:
            return point_data, point_label

    def __len__(self):
        return len(self.datapath)


if __name__ == '__main__':
    
    trainDataset = shapeNetDataset(train=True)
    trainDataLoader = data.DataLoader(trainDataset, batch_size=4)

    for idx, (point_data, point_label) in enumerate(trainDataLoader):
        print(point_data.shape)
from __future__ import print_function, division

import open3d as o3d
import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def visualize_raw_data(filename):
    r"""visualize the raw data
    params:
      The filename of the point cloud data, type='str'
    """
    pointcloud = o3d.io.read_point_cloud(filename, format='xyz')
    o3d.visualization.draw_geometries([pointcloud])


def seg2array(filename):
    r"""Load the .seg file which contains the label, and transform it
    to numpy array
    params:
      The filename of the labels of the points, type='str'
    """
    data = open(filename)
    label = []
    for i in data:
        i = i.strip().split()
        label.append(i)
    return np.asarray(label, dtype=np.int)

def feature_loss(trans_features):
    r"""
    L_{reg} = || I - AA^{T} ||_{F}^{2}
    Loss to constrain the feature transformation matrix 
    to be close to orthogonal matrix
    """
    A = trans_features[0]
    # print("A.shape = ", A.shape)
    I_AAT = torch.eye(A.shape[0]).to(device) - torch.mm(A, A.transpose(0, 1))
    # print(type(I_AAT))
    return (I_AAT * I_AAT).sum()

def class2onehot(classNum):
    onehot = torch.zeros(16)
    onehot[classNum] = 1.0
    return onehot


if __name__ == '__main__':
    filename = BASE_DIR + "/dataset/train_data/02773838/000062.pts"
    file_test = BASE_DIR + "/dataset/train_label/02773838/000062.seg"
    label = seg2array(file_test)
    print(len(label))
    pointcloud = o3d.io.read_point_cloud(filename, format='xyz')
    o3d.visualization.draw_geometries([pointcloud])
    print(len(np.asarray(pointcloud.points)))


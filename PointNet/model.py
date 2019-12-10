from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TNet1(nn.Module):
    r"""
    The ﬁrst transformation network is a mini-PointNet that takes raw 
    point cloud as input and regresses to a 3 × 3 matrix. It’s composed 
    of a shared MLP(64,128,1024) network (with layer output sizes 64, 128, 1024) 
    on each point, a max pooling across points and two fully connected 
    layers with output sizes 512, 256. The output matrix is initialized 
    as an identity matrix. All layers, except the last one, include ReLU 
    and batch normalization.
    """
    def __init__(self, k=3):
        super(TNet1, self).__init__()
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        r"""
        Here x has been transposed
        """
        batch_size = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = torch.max(x, 2, keepdim=True)[0]

        x = x.view(-1, 1024)
        #print(self.fc1(x).size())
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        identity = torch.autograd.Variable(torch.from_numpy(np.eye(self.k, dtype=np.float32)))\
            .view(1, self.k * self.k).repeat(batch_size, 1)

        if x.is_cuda:
            identity = identity.cuda()
        x = x + identity
        x = x.view(-1, 3, 3)
        return x


class TNet2(nn.Module):
    r"""
    The second transformation network has the samea rchitecture
    as the ﬁrst one except that the output is a 64×64 matrix.
    The matrix is also initialized as an identity.
    """
    def __init__(self, k=64):
        super(TNet2, self).__init__()
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        r"""
        Here x has been transposed
        """
        batch_size = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = torch.max(x, 2, keepdim=True)[0]

        x = x.view(-1, 1024)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        identity = torch.autograd.Variable(torch.from_numpy(np.eye(self.k, dtype=np.float32)))\
            .view(1, self.k * self.k).repeat(batch_size, 1)

        if x.is_cuda:
            identity = identity.cuda()
        x = x + identity
        x = x.view(-1, self.k, self.k)
        return x

class global_feature(nn.Module):
    r"""
    Plain network for classification in the paper
    """
    def __init__(self, input_transform=True, feature_transform=True):
        super(global_feature, self).__init__()
        self.in_trans = TNet1()
        self.fea_trans = TNet2()
        self.input_transform = input_transform
        self.feature_transform = feature_transform

        self.conv1 = nn.Conv1d(3, 64, 1)
        # self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

    def forward(self, x):
        # print(x.shape)
        x = x.transpose(1, 2)
        if self.input_transform:
            x = torch.bmm(x.transpose(1, 2), self.in_trans(x))
            x = x.transpose(1, 2)
        # x : shape=(batch, 3, N)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_features = self.fea_trans(x)
            x = torch.bmm(x.transpose(1, 2), trans_features)
            x = x.transpose(1, 2)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        #global feature
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.feature_transform:
            return x, trans_features
        else:
            return x

class PointNetCls(nn.Module):
    def __init__(self, k=16, feature_transform=True):
        super(PointNetCls, self).__init__()

        self.global_feature = global_feature()

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

        self.feature_transform = feature_transform

    def forward(self, x):
        if self.feature_transform:
            x, trans_features = self.global_feature(x)
            # print(x.shape)
            x = F.relu(self.bn1(self.fc1(x)))
            x = F.relu(self.bn2(self.fc2(x)))
            x = self.fc3(x)
            return F.softmax(x, dim=1), trans_features
        else:
            x = self.global_feature(x)
            # rint(x.shape)
            x = F.relu(self.bn1(self.fc1(x)))
            x = F.relu(self.bn2(self.fc2(x)))
            x = self.fc3(x)
            return F.softmax(x, dim=1)
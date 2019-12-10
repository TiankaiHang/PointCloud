from __future__ import print_function, division

import numpy as np
import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dataset
import model
from utils import *

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs("checkpoints", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--num_epochs", type=int, default=10, help="The number of epoches")
parser.add_argument("--batchSize", type=int, default=8, help="The batch size")
parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
parser.add_argument("--checkpoint", type=str, default=None, help="The checkpoint for the model")
parser.add_argument("--dataset", type=str, default=BASE_DIR, help="The root of the dataset")
parser.add_argument("--feature_transform", type=bool, default=True, help="Use feature transform")
parser.add_argument("--shuffle", type=bool, default=True, help="Shuffle or not?")
parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")
parser.add_argument("--sample_interval", type=int, default=1e3, help="The interval to save the model parameters")

opt = parser.parse_args()

batch_size = int(opt.batchSize)
num_workers = int(opt.num_workers)
num_epochs = int(opt.num_epochs)
sample_interval = int(opt.sample_interval)
checkpoint_dir = os.path.join(BASE_DIR, "checkpoints")

trainDataset = dataset.shapeNetDataset(train=True)
trainDataLoader = torch.utils.data.DataLoader(trainDataset,
                                              batch_size=batch_size,
                                              shuffle=opt.shuffle,
                                              num_workers=int(opt.num_workers),
                                              drop_last=True)

testDataset = dataset.shapeNetDataset(train=False)
testDataLoader = torch.utils.data.DataLoader(testDataset,
                                             batch_size=batch_size,
                                             shuffle=opt.shuffle,
                                             num_workers=int(opt.num_workers))

pointNetcls = model.PointNetCls().to(device)

optimizer = optim.Adam(pointNetcls.parameters(), lr=1e-3, betas=(0.9, 0.99))
loss_fn = nn.BCELoss()

lr_schedule = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

if opt.checkpoint:
    pointNetcls.load_state_dict(torch.load(opt.checkpoint))

def train(epochs, trans_feature=True):
    for i in range(epochs):
        lr_schedule.step()
        num_batch = int(len(trainDataset) / batch_size)
        for idx, (data, label) in enumerate(trainDataLoader):
            # print(data.shape, label.shape)
            data, label = data.to(device), label.to(device)

            optimizer.zero_grad()
            pointNetcls.train()
            predict, trans_fea = pointNetcls(data)
            loss = loss_fn(predict, label)
            if trans_feature:
                loss += feature_loss(trans_fea) * 0.001
            loss.backward()
            optimizer.step()

            #print(len(trainDataLoader))
            batch_done = i * len(trainDataLoader) + idx + 1
            if idx % 100 == 0:
                print("[Epoch: {} / {}]  \t [Batch {} / {}] \t loss = {}"\
                    .format(i, epochs, idx, num_batch, loss.item()))

            if sample_interval > 0 \
                and batch_done % sample_interval == 0:
                torch.save(pointNetcls.state_dict(), os.path.join(
                    checkpoint_dir, "train{}.pth".format(batch_done)))

if __name__ == '__main__':
    train(num_epochs)
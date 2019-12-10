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
parser.add_argument("--checkpoint", type=str, default="checkpoints/train10000.pth", 
                    help="The checkpoint for the model")
parser.add_argument("--dataset", type=str, default=BASE_DIR, help="The root of the dataset")
parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")
parser.add_argument("--shuffle", type=bool, default=True, help="Shuffle or not?")

opt = parser.parse_args()

num_workers = int(opt.num_workers)

testDataset = dataset.shapeNetDataset(train=False)
testDataLoader = torch.utils.data.DataLoader(testDataset,
                                             batch_size=2,
                                             shuffle=opt.shuffle,
                                             num_workers=num_workers,
                                             drop_last=True)

pointNetcls = model.PointNetCls().to(device)
pointNetcls.load_state_dict(torch.load(opt.checkpoint))

def test_cls():
    count = 0
    for  i, (data, label) in enumerate(testDataLoader):
        data, label = data.to(device), label.to(device)
        predict = pointNetcls(data)[0]
        # print(predict)
        for j in range(len(predict)):
            #print(predict[j].shape, '\n', label[j].shape)
            predict_cls = torch.argmax(predict[j])
            real_cls = torch.argmax(label[j])
            print("predict: {} \t real: {}".format(predict_cls, real_cls))
            if real_cls == predict_cls:
                count += 1

    return count / len(testDataset)

if __name__ == '__main__':
    accuracy = test_cls()
    print(accuracy)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zhanghuangzhao
"""


import torch
from torch import nn

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # 初始层
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 第二层
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 第三层
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 添加额外的卷积层
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 1 * 1, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = x.reshape((-1, 1, 28, 28))
        x = self.features(x)
        x = self.classifier(x)
        return x


        

if __name__ == "__main__":
    
    from fmnist_dataset import load_fashion_mnist
    from torch.utils.data import DataLoader

    train, dev, test = load_fashion_mnist("../data")
    train_dataloader = DataLoader(train, batch_size=1)
    
    m = CNN()
    
    for x, y in train_dataloader:
        
        l = m(x)
        break
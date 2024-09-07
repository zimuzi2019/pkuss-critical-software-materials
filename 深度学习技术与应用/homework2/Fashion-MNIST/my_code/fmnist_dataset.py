#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zhanghuangzhao
"""

import torch
from torch.utils.data import Dataset
from mnist import MNIST
from torchvision import transforms

class FashionMNISTDataset(Dataset):
    
    def __init__(self, data, transform=None):
        
        x, y = data
        assert len(x) == len(y)
        self.x = x
        self.y = y
        self.transform = transform
        
    def __len__(self):
        
        return len(self.x)
    
    def __getitem__(self, idx):
        image = torch.tensor(self.x[idx]).float().reshape([1,28,28])
        if self.transform:
            image = self.transform(image)
        
        return image, \
                torch.tensor(self.y[idx])
    
def load_fashion_mnist(fmnist_dir, n_dev=10000, random=None):
    
    fmnist = MNIST(fmnist_dir, return_type="lists")
    train = fmnist.load_training()
    test = fmnist.load_testing()
    
    assert n_dev >= 0 and n_dev <= len(train[0]), \
            "Invalid dev size %d, should be within 0 to %d" \
            % (n_dev, len(train[0]))
    if random is None:
        import random
    idx = random.sample(range(len(train[0])), len(train[0]))
    
    
    # Define transforms
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    
    
    dev = [], []
    for i in idx[:n_dev]:
        dev[1].append(train[1][i])
        dev[0].append(train[0][i])
    _train = [], []
    for i in idx[n_dev:]:
        _train[1].append(train[1][i])
        _train[0].append(train[0][i])
    #return FashionMNISTDataset(_train, transform_train), FashionMNISTDataset(dev,transform_test), FashionMNISTDataset(test, transform_test)
    return FashionMNISTDataset(_train), FashionMNISTDataset(dev), FashionMNISTDataset(test)


if __name__ == "__main__":
    
    from torch.utils.data import DataLoader
    train, dev, test = load_fashion_mnist("../data")
    train_dataloader = DataLoader(train, batch_size=1)
    dev_dataloader = DataLoader(dev, batch_size=1)
    test_dataloader = DataLoader(test, batch_size=1)
    
    try:
        import matplotlib.pyplot as plt
        label = 0
        for i, (x, y) in enumerate(train_dataloader):
            if y.item() != label:
                continue
            plt.imshow(x.reshape([28, 28]))
            plt.imsave("../images/fmnist_%d.jpg" % label, x.reshape([28, 28]))
            plt.show()
            label += 1
            if label >= 10:
                break
    except Exception as e:
        print(e)
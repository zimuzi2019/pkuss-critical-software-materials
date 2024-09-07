import torch
from torchvision import models,transforms
from torchvision.datasets import CIFAR10
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader

def cifar10(batch_sz, random_seed, valid_size=0.1, shuffle=True):

  if random_seed is not None:
    torch.manual_seed(random_seed)

  transform_train = transforms.Compose([
      transforms.ToTensor(),
  ])

  #transform_train = transforms.ToTensor()
  transform_valid = transforms.ToTensor()
  transform_test = transforms.ToTensor()

  train_dataset = CIFAR10(root='./datasets', train=True, download=True, transform=transform_train)
  valid_dataset = CIFAR10(root='./datasets', train=True, download=True, transform=transform_valid)

  num_train = len(train_dataset)
  indices = list(range(num_train))
  split = int(np.floor(valid_size * num_train))
  if shuffle:
    np.random.seed(random_seed)
    np.random.shuffle(indices)
  train_idx, valid_idx = indices[split:], indices[:split]

  train_sampler = SubsetRandomSampler(train_idx)
  valid_sampler = SubsetRandomSampler(valid_idx)
  train_loader = DataLoader(train_dataset, batch_size=batch_sz, sampler=train_sampler, pin_memory=True)
  valid_loader = DataLoader(valid_dataset, batch_size=batch_sz, sampler=valid_sampler, pin_memory=True)

  test_dataset = CIFAR10(root='./datasets', train=False, download=True, transform=transform_test)
  test_loader = DataLoader(test_dataset, batch_size=batch_sz)

  return train_loader, valid_loader, test_loader


def cifar10_augmented(batch_sz, valid_size=0.1, shuffle=True, random_seed=42):
  transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(0.1),
    transforms.RandomRotation(5),
    transforms.RandomResizedCrop(32, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
  ])
    
  transform_valid = transforms.Compose([
    transforms.ToTensor(),
  ])

  transform_test = transforms.ToTensor()

  train_dataset = CIFAR10(root='./datasets', train=True, download=True, transform=transform_train)
  valid_dataset = CIFAR10(root='./datasets', train=True, download=True, transform=transform_valid)

  num_train = len(train_dataset)
  indices = list(range(num_train))
  split = int(np.floor(valid_size * num_train))
  if shuffle:
    np.random.seed(random_seed)
    np.random.shuffle(indices)
  train_idx, valid_idx = indices[split:], indices[:split]

  train_sampler = SubsetRandomSampler(train_idx)
  valid_sampler = SubsetRandomSampler(valid_idx)
  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_sz, sampler=train_sampler, pin_memory=True)
  valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_sz, sampler=valid_sampler, pin_memory=True)

  test_dataset = CIFAR10(root='./datasets', train=False, download=True, transform=transform_test)
  test_loader = DataLoader(test_dataset, batch_size=batch_sz)

  return train_loader, valid_loader, test_loader

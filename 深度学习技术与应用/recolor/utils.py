import torch
from torch import nn
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def convert_to_grayscale(batch):

  grayscale_dataset = []
  for image in batch:
    # Convert colored image to grayscale by taking the average of all channels
    grayscale_image = torch.mean(image, dim=0, keepdim=True)
    grayscale_dataset.append((grayscale_image))

  grayscale_dataset = torch.stack(grayscale_dataset, dim=0)
  return grayscale_dataset

def visualize_dataset(dataloader):
  batch = next(iter(dataloader))
  images = batch[0]
  labels = batch[1]
  plt.figure(figsize=(15,15))
  for i in range(32):
    plt.subplot(8,8,i+1)
    img = np.transpose(images[i].numpy(), (1, 2, 0)) 
    plt.imshow(img)
    plt.title(classes[labels[i]])
    plt.axis("off")


def display_image_grid(images, labels, num_rows, num_cols, title_text):

  fig = plt.figure(figsize=(num_cols*2.5, num_rows*2.5), )
  grid = ImageGrid(fig, 111, nrows_ncols=(num_rows, num_cols), axes_pad=0.15)

  for ax, im, l in zip(grid, images, labels):
    if im.size(0) == 1:
      if im.dtype == torch.float32 or im.dtype == torch.float64:
        ax.imshow(np.clip(im.permute(1,2,0).numpy(), 0, 1), cmap = 'gray')
      else:
        ax.imshow(np.clip(im.permute(1,2,0).numpy(), 0, 255), cmap = 'gray')
    else:
      if im.dtype == torch.float32 or im.dtype == torch.float64:
        ax.imshow(np.clip(im.permute(1,2,0).numpy(), 0, 1))
      else:
        ax.imshow(np.clip(im.permute(1,2,0).numpy(), 0, 255))

    ax.axis("off")
    ax.set_title(classes[l])

  plt.suptitle(title_text, fontsize=20)
  plt.tight_layout()
  plt.show()

class Reshape(nn.Module):
  def __init__(self, shape):
    super(Reshape, self).__init__()
    self.shape = shape

  def forward(self, x):
    return x.view(*self.shape)

def weights_init(m):
    """Reinitialize model weights. GAN authors recommend to sampled from N(0,0.2)"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
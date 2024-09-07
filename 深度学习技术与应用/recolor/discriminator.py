import torch
from torch import nn
from torch.nn import functional as F

class Discriminator(nn.Module):
  """Discriminator model"""
  def __init__(self):
    super(Discriminator, self).__init__()
    self.disc_model = nn.Sequential(
        nn.Conv2d(in_channels=4, out_channels=64, kernel_size=5, stride=2, padding=2),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(),

        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(),

        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(256),
        nn.LeakyReLU(),

        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(512),
        nn.LeakyReLU(),


        nn.Flatten(1,-1),
    )

    self.layer_1 = nn.Linear(512 * 2 * 2, 256)
    self.layer_1_b = nn.BatchNorm1d(256)
    self.layer_1_a = nn.LeakyReLU()

    self.layer_2 = nn.Linear(256, 128)
    self.layer_2_b = nn.BatchNorm1d(128)
    self.layer_2_a = nn.LeakyReLU()

    self.layer_3 = nn.Linear(128, 1)
    self.layer_3_a = nn.Sigmoid()

  def forward(self, x, condition):    # pass the gray image and the colored image
                                      # x (batch_size, 3, 32, 32) condtion (batch_size, 1, 32, 32)
    x = torch.cat([x,condition], 1)   # (batch_size, 4, 32, 32)

    x = self.disc_model(x)            # (batch_size, 512 * 2 * 2)

    x = self.layer_1_a(self.layer_1_b(self.layer_1(x)))    # (batch_size, 256)
    x = self.layer_2_a(self.layer_2_b(self.layer_2(x)))    # (batch_size, 128)
    x = self.layer_3_a(self.layer_3(x))                    # (batch_size, 1)

    return x
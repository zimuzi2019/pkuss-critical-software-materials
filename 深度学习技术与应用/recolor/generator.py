import torch
from torch import nn
from torch.nn import functional as F


class Generator(nn.Module):
  """Generator model"""
  def __init__(self):
    super(Generator, self).__init__()

    # U-Net architecture

    self.encoding_unit1 = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(),
    )

    self.encoding_unit2 = nn.Sequential(
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(),
        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(),
    )

    self.encoding_unit3 = nn.Sequential(
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(256),
        nn.LeakyReLU(),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(256),
        nn.LeakyReLU(),
    )

    self.encoding_unit4 = nn.Sequential(
        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(512),
        nn.LeakyReLU(),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(512),
        nn.LeakyReLU(),
    )

    self.encoding_unit5 = nn.Sequential(
        nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(1024),
        nn.LeakyReLU(),
        nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(1024),
        nn.LeakyReLU(),
    )

    self.decoding_unit1 = nn.Sequential(
        nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(512),
        nn.LeakyReLU(),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(512),
        nn.LeakyReLU(),
    )

    self.decoding_unit2 = nn.Sequential(
        nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(256),
        nn.LeakyReLU(),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(256),
        nn.LeakyReLU(),
    )

    self.decoding_unit3 = nn.Sequential(
        nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(),
        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(),
    )

    self.decoding_unit4 = nn.Sequential(
        nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(),
    )
    self.decoding_unit5 = nn.Sequential(
        nn.Conv2d(in_channels=64, out_channels=3, kernel_size=1, stride=1, padding=0),
    )

    self.upsampling1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=3, stride=2, padding=1, output_padding=1)
    self.upsampling2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1)
    self.upsampling3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
    self.upsampling4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)

  def forward(self, x):

    x1 = self.encoding_unit1(x)
    x = F.max_pool2d(x1, (2,2))

    x2 = self.encoding_unit2(x)
    x = F.max_pool2d(x2, (2,2))

    x3 = self.encoding_unit3(x)
    x = F.max_pool2d(x3, (2,2))

    x4 = self.encoding_unit4(x)
    x = F.max_pool2d(x4, (2,2))

    x = self.encoding_unit5(x)

    x = self.upsampling1(x)
    x = torch.cat([x, x4], dim=1)
    x = self.decoding_unit1(x)

    x = self.upsampling2(x)
    x = torch.cat([x, x3], dim=1)
    x = self.decoding_unit2(x)

    x = self.upsampling3(x)
    x = torch.cat([x, x2], dim=1)
    x = self.decoding_unit3(x)

    x = self.upsampling4(x)
    x = torch.cat([x, x1], dim=1)
    x = self.decoding_unit4(x)

    x = self.decoding_unit5(x)

    x = F.tanh(x)

    return x
import torch
from torch import nn
import torch.nn.functional as F

class UNet(nn.Module):
  def __init__(self):
    super().__init__()
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

  def encoder(self, x):

    x1 = self.encoding_unit1(x)
    x = F.max_pool2d(x1, (2,2))

    x2 = self.encoding_unit2(x)
    x = F.max_pool2d(x2, (2,2))

    x3 = self.encoding_unit3(x)
    x = F.max_pool2d(x3, (2,2))

    x4 = self.encoding_unit4(x)
    x = F.max_pool2d(x4, (2,2))

    x = self.encoding_unit5(x)

    return x, x1, x2, x3, x4

  def decoder(self, x, x1, x2, x3, x4):    # x (1024, 1, 1), x1 (64, 32, 32), x2 (128, 16, 16), x3 (256, 8, 8), x4 (512, 4, 4)

    x = self.upsampling1(x)                # x (512, 4, 4)
    x = torch.cat([x, x4], dim=1)          # x (1024, 4, 4)  注意别漏了 b 维度，是对 c 进行拼接
    x = self.decoding_unit1(x)             # x (512, 4, 4)

    x = self.upsampling2(x)                # x (256, 8, 8)
    x = torch.cat([x, x3], dim=1)          # x (512, 8, 8)
    x = self.decoding_unit2(x)             # x (256, 8, 8)

    x = self.upsampling3(x)                # x (128, 16, 16)
    x = torch.cat([x, x2], dim=1)          # x (256, 16, 16)
    x = self.decoding_unit3(x)             # x (128, 16, 16)

    x = self.upsampling4(x)
    x = torch.cat([x, x1], dim=1)
    x = self.decoding_unit4(x)

    x = self.decoding_unit5(x)

    return x

  def forward(self, x):                          # x (1, 32, 32)
    x, x1, x2, x3, x4 = self.encoder(x)          # x (1024, 1, 1), x1 (64, 32, 32), x2 (128, 16, 16), x3 (256, 8, 8), x4 (512, 4, 4)

    xhat = self.decoder(x, x1, x2, x3, x4)       # xhat (3, 32, 32)

    return xhat
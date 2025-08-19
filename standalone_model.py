import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np

# Ensure reproducibility
random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""==========================CNN Architecture==============================="""


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=5, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=5, padding=2, bias=True),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=3, padding=1,
                      bias=True)  # outputs residual
        )

    def forward(self, x):
        inp = x
        res = self.decoder(self.encoder(x))
        out = inp + res
        return out.clamp(-1, 1)


"""==========================UNet Architecture==============================="""


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_dropout=False, norm='group'):
        super().__init__()
        if norm == 'instance':
            def Norm(c): return nn.InstanceNorm2d(c, affine=True)
        elif norm == 'group':
            def Norm(c): return nn.GroupNorm(32 if c >= 32 else 1, c)
        elif norm == 'batch':
            def Norm(c): return nn.BatchNorm2d(c)
        else:
            raise ValueError(f"Unknown norm: {norm}")

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.n1 = Norm(out_ch)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.n2 = Norm(out_ch)
        self.relu2 = nn.ReLU(inplace=True)

        self.dropout = nn.Dropout2d(0.2) if use_dropout else nn.Identity()

    def forward(self, x):
        x = self.relu1(self.n1(self.conv1(x)))
        x = self.dropout(x)
        x = self.relu2(self.n2(self.conv2(x)))
        return x


class Encoder(nn.Module):
    def __init__(self, in_ch, out_ch, use_dropout=False, norm='group'):
        super().__init__()
        self.block = ConvBlock(in_ch, out_ch, use_dropout, norm=norm)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        feat = self.block(x)
        down = self.pool(feat)
        return feat, down


class Decoder(nn.Module):
    def __init__(self, up_in, skip_in, out_channels, use_dropout=False, norm='group'):
        super().__init__()
        self.upconv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(up_in, out_channels, 3, padding=1, bias=False)
        )
        self.conv = ConvBlock(out_channels + skip_in,
                              out_channels, use_dropout, norm=norm)

    def forward(self, x1, x2):  # x1: deeper, x2: skip
        x1_up = self.upconv(x1)
        if x1_up.size()[2:] != x2.size()[2:]:
            x1_up = F.interpolate(x1_up, size=x2.size()[
                                  2:], mode='bilinear', align_corners=False)
        x = torch.cat([x2, x1_up], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, use_dropout=False, norm='group', resize_output=False):
        super().__init__()
        self.resize_output = resize_output
        self.in_conv = ConvBlock(in_channels, 64, use_dropout, norm=norm)
        self.enc1 = Encoder(64, 128, use_dropout, norm=norm)
        self.enc2 = Encoder(128, 256, use_dropout, norm=norm)
        self.enc3 = Encoder(256, 512, use_dropout, norm=norm)
        self.enc4 = Encoder(512, 1024, use_dropout, norm=norm)

        self.dec1 = Decoder(1024, 512, 512, use_dropout, norm=norm)
        self.dec2 = Decoder(512, 256, 256, use_dropout, norm=norm)
        self.dec3 = Decoder(256, 128, 128, use_dropout, norm=norm)
        self.dec4 = Decoder(128, 64, 64, use_dropout, norm=norm)

        self.out_conv = nn.Conv2d(64, out_channels, 1)

    def forward(self, inp, target_size=None):
        x1 = self.in_conv(inp)             # H, W
        f2, x2 = self.enc1(x1)             # f2: H/1, x2: H/2
        f3, x3 = self.enc2(x2)             # f3: H/2, x3: H/4
        f4, x4 = self.enc3(x3)             # f4: H/4, x4: H/8
        f5, x5 = self.enc4(x4)             # f5: H/8, x5: H/16

        x = self.dec1(x5, f4)
        x = self.dec2(x, f3)
        x = self.dec3(x, f2)
        x = self.dec4(x, x1)

        y = self.out_conv(x)
        y = inp + y                         # global residual
        if self.resize_output:
            target_size = target_size or x1.shape[2:]
            y = F.interpolate(y, size=target_size,
                              mode='bilinear', align_corners=False)
        return y.clamp(-1, 1)


"""==========================ResNet Architecture==============================="""


class ResBlock(nn.Module):
    def __init__(self, c, norm='group'):
        super().__init__()
        Norm = {'instance': nn.InstanceNorm2d,
                'batch': nn.BatchNorm2d,
                'group': lambda ch: nn.GroupNorm(32 if ch >= 32 else 1, ch)}[norm]
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(c, c, 3, bias=False),
            Norm(c) if norm != 'group' else Norm(c),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(c, c, 3, bias=False),
            Norm(c) if norm != 'group' else Norm(c),
        )

    def forward(self, x): return x + self.block(x)


class ResNet(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, num_resnet_blocks=9, norm='group'):
        super().__init__()
        Norm = {'instance': nn.InstanceNorm2d,
                'batch': nn.BatchNorm2d,
                'group': lambda ch: nn.GroupNorm(32 if ch >= 32 else 1, ch)}[norm]

        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, kernel_size=7, bias=False),
            Norm(64) if norm != 'group' else Norm(64),
            nn.ReLU(inplace=True)
        ]

        in_ch = 64
        for _ in range(2):
            out_ch = in_ch * 2
            layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3,
                          stride=2, padding=1, bias=False),
                Norm(out_ch) if norm != 'group' else Norm(out_ch),
                nn.ReLU(inplace=True)
            ]
            in_ch = out_ch

        for _ in range(num_resnet_blocks):
            layers += [ResBlock(in_ch, norm=norm)]

        for _ in range(2):
            out_ch = in_ch // 2
            layers += [
                nn.Upsample(scale_factor=2, mode='bilinear',
                            align_corners=False),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                Norm(out_ch) if norm != 'group' else Norm(out_ch),
                nn.ReLU(inplace=True)
            ]
            in_ch = out_ch

        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_nc, kernel_size=7, bias=True)
        ]
        self.body = nn.Sequential(*layers)

        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        y = x + self.body(x)   # global residual
        return y.clamp(-1, 1)

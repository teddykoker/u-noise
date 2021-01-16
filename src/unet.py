"""
Unet model architecture.
Used in segmentation and image noising.
"""
import torch
import torch.nn as nn


def _conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv = _conv_block(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net: Convolutional Networks for Biomedical Image Segmentation
    https://arxiv.org/abs/1505.04597

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        depth (int): depth of network
        cf (int): Channel factor, number of channels for first layers output is 2 ** cf
    """

    def __init__(self, in_channels=3, out_channels=1, depth=5, cf=6):
        super(UNet, self).__init__()
        self.depth = depth
        self.downs = nn.ModuleList(
            [
                _conv_block(
                    in_channels=(in_channels if i == 0 else 2 ** (cf + i - 1)),
                    out_channels=(2 ** (cf + i)),
                )
                for i in range(depth)
            ]
        )
        self.ups = nn.ModuleList(
            [
                Up(in_channels=(2 ** (cf + i + 1)), out_channels=(2 ** (cf + i)))
                for i in reversed(range(depth - 1))
            ]
        )
        self.max = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1x1 = nn.Conv2d(2 ** cf, out_channels, kernel_size=1)

    def forward(self, x):
        outs = []
        for i, down in enumerate(self.downs):
            x = down(x)
            if i != (self.depth - 1):
                outs.append(x)
                x = self.max(x)

        for i, up in enumerate(self.ups):
            x = up(x, outs[-i - 1])

        return self.conv1x1(x)

import torch
import torch.nn as nn
from timm.models.layers import DropPath
from torch import Tensor
import torch.nn.functional as F
import math


class Stem(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=4, stride=4)
        self.norm = nn.BatchNorm2d(output_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x


class GFAEncoder(nn.Module):
    """
    GFAEncoder class represents the encoder module of the GFA (Global Feature Aggregation) network.
    It applies a series of operations including normalization, attention, convolution, and MLP to the input tensor.

    Args:
        dim (int): The input dimension of the tensor.
        feat_size (int): The size of the input feature map.
        drop_path (float, optional): The dropout probability for the DropPath operation. Defaults to 0.0.
        expan_ratio (int, optional): The expansion ratio for the pointwise convolutions. Defaults to 4.
        n_heads (int, optional): The number of attention heads. Defaults to 4.
        qkv_bias (bool, optional): Whether to include bias in the qkv linear layer. Defaults to True.
    """

    def __init__(self, dim, feat_size, drop_path=0.0, expan_ratio=4, n_heads=4, qkv_bias=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.n_heads = n_heads
        self.temp = nn.Parameter(torch.ones(n_heads, 1, 1))
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.dk_ratio = 0.2
        self.dw = nn.Conv2d(dim, dim, kernel_size=3, padding=3 // 2, groups=dim)
        self.avgpool = nn.AvgPool2d(2, stride=2)
        if feat_size % 2 == 0:
            self.convtranspose2d = nn.ConvTranspose2d(dim, dim, 2, stride=2)
        else:
            self.convtranspose2d = nn.ConvTranspose2d(
                dim, dim, kernel_size=2, stride=2, output_padding=1
            )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.pw1 = nn.Linear(dim, expan_ratio * dim)  # pointwise/1x1 convs
        self.act = nn.GELU()
        self.pw2 = nn.Linear(expan_ratio * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.avgpool(self.dw(x) + input)
        B, C, H, W = x.shape
        x = self.norm1(x.reshape(B, C, H * W).permute(0, 2, 1))

        # Attention
        qkv = self.qkv(x).reshape(B, H * W, 3, self.n_heads, C // self.n_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q = torch.nn.functional.normalize(qkv[0].transpose(-2, -1), dim=-1)
        k = torch.nn.functional.normalize(qkv[1].transpose(-2, -1), dim=-1)
        v = qkv[2].transpose(-2, -1)

        # DropKey
        attn = (q @ k.transpose(-2, -1)) * self.temp
        m_c = torch.ones_like(attn) * self.dk_ratio
        attn = attn + torch.bernoulli(m_c) * -1e12
        attn = attn.softmax(dim=-1)
        x = self.proj((attn @ v).permute(0, 3, 1, 2).reshape(B, H * W, C))
        x = (x + self.drop_path(x)).permute(0, 2, 1).reshape(B, C, H, W)

        # convtranspose
        x = self.convtranspose2d(x).permute(0, 2, 3, 1)

        # MLP
        x = self.pw2(self.act(self.pw1(self.norm2(x)))).permute(0, 3, 1, 2)
        x = input + self.drop_path(x)
        return x


class Pconv(nn.Module):
    # Pconv : https://arxiv.org/abs/2303.03667
    def __init__(self, dim, n_div, forward, kernel_size):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        # Adjust padding based on the kernel size to maintain the spatial dimensions
        padding = (kernel_size - 1) // 2
        self.partial_conv3 = nn.Conv2d(
            self.dim_conv3,
            self.dim_conv3,
            kernel_size,
            stride=1,
            padding=padding,
            bias=False,
        )

        if forward == "slicing":
            self.forward = self.forward_slicing
        elif forward == "split_cat":
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x: torch.Tensor) -> torch.Tensor:
        # only for inference
        x = x.clone()  # Clone to keep the original input intact
        x[:, : self.dim_conv3, :, :] = self.partial_conv3(x[:, : self.dim_conv3, :, :])
        return x

    def forward_split_cat(self, x: torch.Tensor) -> torch.Tensor:
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)
        return x


class SE_Module(nn.Module):
    # SE_Module : https://arxiv.org/abs/1709.01507
    def __init__(self, channel, reduction=16):
        super(SE_Module, self).__init__()
        # Define the squeeze operation (global average pooling)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # Define the excitation operation: two fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze
        y = self.avg_pool(x).view(b, c)
        # Excitation
        y = self.fc(y).view(b, c, 1, 1)
        # Scale
        return x * y.expand_as(x)


class LFAEncoder(nn.Module):
    """
    LFAEncoder is a module that performs encoding using a combination of depthwise convolution, pointwise convolution,
    layer normalization, GELU activation, and squeeze-and-excitation module.

    Args:
        dim (int): The number of input channels.
        drop_path (float, optional): The probability of dropping a path during training. Defaults to 0.0.
        expan_ratio (float, optional): The expansion ratio for the pointwise convolution. Defaults to 4.0.
        train (bool, optional): Whether the module is in training mode or not. Defaults to True.
        kernel_s (int, optional): The kernel size for the pointwise convolution. Defaults to 7.
    """

    def __init__(self, dim, drop_path=0.0, expan_ratio=4.0, train=True, kernel_s=7):
        super().__init__()
        self.dwconv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=3 // 2, groups=dim)
        
        if train:
            self.dwconv2 = Pconv(
                dim=dim, n_div=4, forward="split_cat", kernel_size=kernel_s
            )
        else:
            self.dwconv2 = Pconv(
                dim=dim, n_div=4, forward="slicing", kernel_size=kernel_s
            )

        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, int(expan_ratio * dim))
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(int(expan_ratio * dim), dim)
        self.se = SE_Module(channel=dim, reduction=16)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv2(self.dwconv1(x) + input)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv2(self.act(self.pwconv1(self.norm(x))))
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = self.se(input + self.drop_path(x))
        return x


class DownsampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride
        )

    def forward(self, x):
        return self.conv(x)


def make_stage(in_channels, out_channels, lfae_depth, lfae_kernel_size, gfae_depth, feat_size, train, drop_path=0.0):
    layers = [DownsampleLayer(in_channels, out_channels)]
    for _ in range(lfae_depth):
        layers.append(
            LFAEncoder(dim=out_channels, kernel_s=lfae_kernel_size, train=train, drop_path=drop_path)
        )
    for _ in range(gfae_depth):
        layers.append(
            GFAEncoder(dim=out_channels, feat_size=feat_size, drop_path=drop_path)
        )
    return nn.Sequential(*layers)


class MixMobileNet(nn.Module):
    def __init__(self, variant="XXS", img_size=256, num_classes=1000, train=True):
        super().__init__()
        # Define configurations for each variant
        configs = {
            "XXS": {
                "channels": [24, 48, 88, 168],
                "lfae_depth": [2, 1, 5, 1],
                "lfae_kernel_size": [3, 5, 7, 9],
                "gfae_depth": [0, 1, 1, 1],
            },
            "XS": {
                "channels": [32, 64, 100, 192],
                "lfae_depth": [3, 2, 8, 2],
                "lfae_kernel_size": [3, 5, 7, 9],
                "gfae_depth": [0, 1, 1, 1],
            },
            "S": {
                "channels": [48, 96, 160, 304],
                "lfae_depth": [3, 2, 8, 2],
                "lfae_kernel_size": [3, 5, 7, 9],
                "gfae_depth": [0, 1, 1, 1],
            },
            "M": { # Experimental
                "channels": [32, 64, 128, 256],
                "lfae_depth": [3, 2, 9, 2],
                "lfae_kernel_size": [3, 5, 7, 9],
                "gfae_depth": [0, 1, 3, 1],
            },
        }
        config = configs[variant]
        self.num_features = config["channels"][-1]
        self.num_classes = num_classes

        self.img_size = math.floor(math.sqrt(img_size))

        self.stem = Stem(input_channels=3, output_channels=config["channels"][0])
        self.stages = nn.ModuleList()

        # Instantiate stages with the configured number of channels and blocks
        in_channels = config["channels"][0]
        for i, out_channels in enumerate(config["channels"]):
            stage = make_stage(
                in_channels=in_channels,
                out_channels=out_channels,
                lfae_depth=config["lfae_depth"][i],
                lfae_kernel_size=config["lfae_kernel_size"][i],
                gfae_depth=config["gfae_depth"][i],
                feat_size=self.img_size,
                train=train,
                drop_path=0.0,  # Placeholder for drop_path, adjust as needed
            )
            self.stages.append(stage)
            in_channels = out_channels
            if config["gfae_depth"][i] > 0:
                self.img_size = self.img_size // 2

        if num_classes > 0:
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(in_channels, num_classes)
            )
        else:
            self.head = nn.Identity()

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.head(x)
        return x

import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from torch import Tensor
import torch.nn.functional as F
import math
import numpy as np


class Residual(nn.Module):
    def __init__(self, fn1):
        super().__init__()
        self.fn1 = fn1
    def forward(self, x, **kwargs):
        return self.fn1(x, **kwargs) + x

class SpectralGatingNetwork(nn.Module):
    def __init__(self, dim, h=14, w=8):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02)
        self.w = w
        self.h = h

    def forward(self, x, spatial_size=None):
        B, C, H, W = x.shape
        
        if spatial_size is None:
            a, b = H, W  # Directly using H and W as spatial dimensions
        else:
            a, b = spatial_size

        x = x.permute(0, 2, 3, 1).contiguous()  # New shape B, H, W, C
        
        x = x.to(torch.float32)
        
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(a, b), dim=(1, 2), norm='ortho')
        
        # Before reshaping back, transpose x from B, H, W, C back to B, C, H, W
        x = x.permute(0, 3, 1, 2).contiguous()  # Adjusted shape back to B, C, H, W
        
        return x


class Stem(nn.Module):
    def __init__(self, input_channels, output_channels, spect, h, w):
        super().__init__()
        self.spect = spect
        if self.spect:
            self.spectral = Residual(SpectralGatingNetwork(input_channels, h, w))
            self.pos_embed_spectral = nn.Parameter(torch.zeros(1, input_channels, h, h))
            self._trunc_normal_(self.pos_embed_gfae, std=.02)
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=4, stride=4)
        self.norm = nn.BatchNorm2d(output_channels)

    def forward(self, x):
        if self.spect:
            x = self.spectral(x + self.pos_embed_spectral)
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

    def __init__(self, dim, drop_path=0.0, expan_ratio=4, n_heads=4, qkv_bias=True):
        super().__init__()

        self.n_heads = n_heads

        self.avgpool = nn.AvgPool2d(2, stride=2)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        
        self.temp = nn.Parameter(torch.ones(n_heads, 1, 1))
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.dk_ratio = 0.2
        self.dw = nn.Conv2d(dim, dim, kernel_size=3, padding=3 // 2, groups=dim)
        
        self.convtranspose2d = nn.ConvTranspose2d(dim, dim, 2, stride=2)
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
            in_channels, out_channels, kernel_size=kernel_size, stride=stride,
        )

    def forward(self, x):
        x = self.conv(x)
        # Check if the height or width is odd, and if so, pad to make it even
        height, width = x.size()[2:]
        pad_height = (height % 2 != 0)
        pad_width = (width % 2 != 0)
        if pad_height or pad_width:
            # Apply padding to the bottom and/or right if needed
            x = F.pad(x, (0, pad_width, 0, pad_height))
        return x


class PositionalEncoding2D(nn.Module):
    def __init__(self, channels: int, max_size:  int = 100, dropout: float = 0.0):
        super(PositionalEncoding2D, self).__init__()
        self.channels = channels
        self.dropout = nn.Dropout(p=dropout)
        adjusted_channels = int(np.ceil(channels / 4) * 2)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, adjusted_channels, 2).float() / adjusted_channels))

        # Precompute positional encodings for max_size in both dimensions
        pos = torch.arange(max_size).unsqueeze(1)
        sin_inp = torch.einsum("i,j->ij", pos.flatten(), inv_freq)

        emb_sin = torch.sin(sin_inp)
        emb_cos = torch.cos(sin_inp)
        emb = torch.cat((emb_sin, emb_cos), dim=-1)

        # Ensure the embedding size matches the channel size exactly
        if self.channels % 2 == 1:  # Odd number of channels
            emb = torch.cat((emb, emb[:, :1]), dim=-1)  # Append one more column to match size

        self.register_buffer('emb', emb)

    def forward(self, tensor):
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")

        _, orig_ch, x, y = tensor.shape
        emb_x = self.emb[:x, :self.channels//2].unsqueeze(1).repeat(1, y, 1)
        emb_y = self.emb[:y, :self.channels//2].unsqueeze(0).repeat(x, 1, 1)

        # Corrected line for creating an empty tensor with the proper device and dtype
        emb = torch.empty(x, y, self.channels, device=tensor.device, dtype=tensor.dtype)
        emb[..., ::2] = emb_x
        emb[..., 1::2] = emb_y

        # Adjust if the number of channels in the input tensor is different
        if orig_ch != self.channels:
            if orig_ch < self.channels:
                emb = emb[..., :orig_ch]
            else:  # If more channels in input than embeddings, replicate the last channel
                emb = torch.cat([emb, emb[..., -1:]], dim=-1)
        emb = emb.permute(2, 0, 1)
        tensor = tensor + emb[None, :, :, :]
        return self.dropout(tensor)


class MixMobileBlock(nn.Module):
    def __init__(self, in_channels, out_channels, lfae_depth, lfae_kernel_size, gfae_depth, feat_size, train, add_pos, learnable_pos,
                drop_path=0.0, dropout=0.0):
        super().__init__()
        self.downsample = DownsampleLayer(in_channels, out_channels)
        self.lfae_depth = lfae_depth
        self.gfae_depth = gfae_depth
        self.add_pos = add_pos
        self.learnable_pos = learnable_pos

        assert not (add_pos and learnable_pos), "add_pos and learnable_pos cannot both be True"

        if add_pos:
            self.pos_embed_lfae = PositionalEncoding2D(channels=out_channels, max_size=feat_size, dropout=dropout)
            if self.gfae_depth > 0:
                self.pos_embed_gfae = PositionalEncoding2D(channels=out_channels, max_size=feat_size, dropout=dropout)

        if learnable_pos:
            self.pos_embed_lfae = nn.Parameter(torch.zeros(1, out_channels, feat_size, feat_size))
            self._trunc_normal_(self.pos_embed_lfae, std=.02)
            if self.gfae_depth > 0:
                self.pos_embed_gfae = nn.Parameter(torch.zeros(1, out_channels, feat_size, feat_size))
                self._trunc_normal_(self.pos_embed_gfae, std=.02)

        self.lfae_layers = nn.ModuleList([
            Residual(LFAEncoder(dim=out_channels, kernel_s=lfae_kernel_size, train=train, drop_path=drop_path))
            for _ in range(lfae_depth)
        ])
        
        if gfae_depth > 0:
            self.gfae_layers = nn.ModuleList([
                Residual(GFAEncoder(dim=out_channels, drop_path=drop_path))
                for _ in range(gfae_depth)
            ])
        else:
            self.gfae_layers = nn.Identity()
    
    def _trunc_normal_(self, tensor, mean=0., std=1.):
        trunc_normal_(tensor, mean=mean, std=std)

    def forward(self, x):
        if self.gfae_depth > 0:
            x = self.downsample(x)
        
        if self.add_pos:
            x = self.pos_embed_lfae(x)
        if self.learnable_pos:
            x = x + self.pos_embed_lfae
        for lfae in self.lfae_layers:
            x = lfae(x)
        
        if self.gfae_depth > 0:
            if self.add_pos:
                x = self.pos_embed_gfae(x)
            if self.learnable_pos:
                x = x + self.pos_embed_gfae
            for gfae in self.gfae_layers:
                x = gfae(x)
        return x


class MixMobileNet(nn.Module):
    def __init__(self, variant="XXS", img_size=256, num_classes=1000, train=True, add_pos=True, learnable_pos=False,
                 init_weights=False, spect=False, drop_path=0.0, dropout=0.0):
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
            "SS": { # Experimental
                "channels": [32, 64, 128, 256],
                "lfae_depth": [3, 2, 9, 2],
                "lfae_kernel_size": [3, 5, 7, 9],
                "gfae_depth": [0, 1, 3, 1],
            },
            "M": { # Experimental
                "channels": [64, 128, 256, 512],
                "lfae_depth": [3, 2, 9, 2],
                "lfae_kernel_size": [3, 5, 7, 9],
                "gfae_depth": [0, 1, 3, 1],
            },
            "L": { # Experimental
                "channels": [128, 256, 512, 1024],
                "lfae_depth": [3, 2, 14, 2],
                "lfae_kernel_size": [3, 5, 7, 9],
                "gfae_depth": [0, 1, 13, 1],
            },
        }
        config = configs[variant]
        self.num_features = config["channels"][-1]
        self.num_classes = num_classes

        # self.img_size = math.floor(math.sqrt(img_size))
        self.img_size = img_size // 4

        self.stem = Stem(input_channels=3, output_channels=config["channels"][0], spect, img_size, img_size / 2 + 1)
        self.stages = nn.ModuleList()

        # Instantiate stages with the configured number of channels and blocks
        in_channels = config["channels"][0]
        for i, out_channels in enumerate(config["channels"]):
            mmb = MixMobileBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                lfae_depth=config["lfae_depth"][i],
                lfae_kernel_size=config["lfae_kernel_size"][i],
                gfae_depth=config["gfae_depth"][i],
                feat_size=self.img_size,
                train=train,
                add_pos=add_pos,
                learnable_pos=learnable_pos,
                drop_path=drop_path,
                dropout=dropout,
            )
            self.stages.append(mmb)
            in_channels = out_channels
            self.img_size = (self.img_size  // 2) + (1 if (self.img_size  // 2) % 2 != 0 else 0)

        if num_classes > 0:
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(in_channels, num_classes)
            )
        else:
            self.head = nn.Identity()

        if init_weights:
            self.apply(self._init_weights)

    def _trunc_normal_(self, tensor, mean=0., std=1.):
        trunc_normal_(tensor, mean=mean, std=std)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            self._trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            self._trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.head(x)
        return x

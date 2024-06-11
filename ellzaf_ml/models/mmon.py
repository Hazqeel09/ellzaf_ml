import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
import torch.nn.functional as F
import numpy as np
from collections import namedtuple
from packaging import version
from typing import Optional, List, Tuple
import math
import torch.nn.init as init

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


Config = namedtuple('FlashAttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])


class DualFunc(nn.Module):
    def __init__(self, fn1, fn2):
        super().__init__()
        self.fn1 = fn1
        self.fn2 = fn2
    def forward(self, x, **kwargs):
        return self.fn2(self.fn1(x, **kwargs), **kwargs) + x


class Stem(nn.Module):
    def __init__(self, input_channels, output_channels, spect, h, w):
        super().__init__()
        self.spect = spect
        if self.spect:
            self.pos_embed_spectral = nn.Parameter(torch.zeros(1, input_channels, h, h))
            self._trunc_normal_(self.pos_embed_spectral, std=.02)
            self.spectral = SpectralGatingNetwork(input_channels, h, w)
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=4, stride=4)
        self.norm = nn.BatchNorm2d(output_channels)

    def _trunc_normal_(self, tensor, mean=0., std=1.):
        trunc_normal_(tensor, mean=mean, std=std)

    def forward(self, x):
        if self.spect:
            x = self.spectral(x + self.pos_embed_spectral)
        x = self.conv(x)
        x = self.norm(x)
        return x

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU()
                    )
    
    def forward(self, x: torch.Tensor):
        return self.conv(x)

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, out_channels=96, h=14, w=8, spect=False):
        super().__init__()
        self.spect = spect
        if self.spect:
            self.pos_embed_spectral = nn.Parameter(torch.zeros(1, in_channels, h, h))
            self._trunc_normal_(self.pos_embed_spectral, std=.02)
            self.spectral = SpectralGatingNetwork(in_channels, h, w)
        self.conv1 = BasicBlock(in_channels, out_channels//2, 3, 2)
        self.conv2 = BasicBlock(out_channels//2, out_channels, 3, 2)
        self.conv3 = BasicBlock(out_channels, out_channels, 3, 1)
        self.conv4 = BasicBlock(out_channels, out_channels, 3, 1)
        self.conv5 = nn.Conv2d(out_channels, out_channels, 1, 1, 0)
        self.layernorm = nn.GroupNorm(1, out_channels)

    def _trunc_normal_(self, tensor, mean=0., std=1.):
        trunc_normal_(tensor, mean=mean, std=std)

    def forward(self, x: torch.Tensor):
        if self.spect:
            x = self.spectral(x + self.pos_embed_spectral)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return self.layernorm(x)

class SpectralGatingNetwork(nn.Module):
    def __init__(self, dim, h=14, w=8):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(int(h), int(w), dim, 2, dtype=torch.float32) * 0.02)
        self._trunc_normal_(self.complex_weight, std=.02)
        self.w = w
        self.h = h

    def _trunc_normal_(self, tensor, mean=0., std=1.):
        trunc_normal_(tensor, mean=mean, std=std)

    def forward(self, x, spatial_size=None):
        _, _, H, W = x.shape
        
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
    

class Attend(nn.Module):
    def __init__(self, use_flash = False):
        super().__init__()
        assert not (use_flash and version.parse(torch.__version__) < version.parse('2.0.0')), 'in order to use flash attention, you must be using pytorch 2.0 or above'

        # determine efficient attention configs for cuda and cpu

        self.cpu_config = Config(True, True, True)
        self.cuda_config = None

        if not torch.cuda.is_available():
            return

        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))

        if device_properties.major == 8 and device_properties.minor == 0:
            self.cuda_config = Config(True, False, False)
        else:
            self.cuda_config = Config(False, True, True)

    def flash_attn(self, q, k, v):
        config = self.cuda_config if q.is_cuda else self.cpu_config

        # flash attention - https://arxiv.org/abs/2205.14135
        
        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(q, k, v)

        return out

    def forward(self, q, k, v):
        return self.flash_attn(q, k, v)


class GFAEncoder(nn.Module):
    def __init__(self, dim, dropout=0.0,drop_path=0.0, expan_ratio=4, dk_ratio=0.2, n_heads=4, use_flash=False, avgpool=True, qkv_bias=True):
        super().__init__()

        self.n_heads = n_heads
        self.use_flash = use_flash
        self.avgpool = avgpool

        self.dw = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        if self.avgpool:
            self.pool = nn.AvgPool2d(2, stride=2)
        else:
            self.pool = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, stride=2)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        if not self.use_flash:
            self.temp = nn.Parameter(torch.ones(n_heads, 1, 1))
            self.proj = nn.Linear(dim, dim)
            self.dk_ratio = dk_ratio
            self.dropout1 = nn.Dropout(p=dropout)
        else:
            self.attend = Attend(use_flash)
        
        self.convtranspose2d = nn.ConvTranspose2d(dim, dim, 2, stride=2)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.pw1 = nn.Linear(dim, expan_ratio * dim)  # pointwise/1x1 convs
        self.act = nn.GELU()
        self.dropout2 = nn.Dropout(p=dropout)
        self.pw2 = nn.Linear(expan_ratio * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.pool(self.dw(x) + input)
        B, C, H, W = x.shape
        x = self.norm1(x.reshape(B, C, H * W).permute(0, 2, 1))

        # Attention
        qkv = self.qkv(x).reshape(B, H * W, 3, self.n_heads, C // self.n_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        if self.use_flash:
            x = self.attend(qkv[0], qkv[1], qkv[2]).reshape(B, H * W, C)
        else:
            q = torch.nn.functional.normalize(qkv[0].transpose(-2, -1), dim=-1)
            k = torch.nn.functional.normalize(qkv[1].transpose(-2, -1), dim=-1)
            v = qkv[2].transpose(-2, -1)

            # DropKey
            attn = (q @ k.transpose(-2, -1)) * self.temp
            m_c = torch.ones_like(attn) * self.dk_ratio
            attn = attn + torch.bernoulli(m_c) * -1e12
            attn = attn.softmax(dim=-1)
            attn = self.proj((attn @ v).permute(0, 3, 1, 2).reshape(B, H * W, C))
            attn = self.dropout1(attn)

        x = (x + self.drop_path(attn)).permute(0, 2, 1).reshape(B, C, H, W)

        # convtranspose
        x = self.convtranspose2d(x).permute(0, 2, 3, 1)

        # MLP
        x = self.pw2(self.dropout2(self.act(self.pw1(self.norm2(x))))).permute(0, 3, 1, 2)
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
    def __init__(self, dim, dropout=0.0, drop_path=0.0, expan_ratio=4.0, train=True, kernel_s=7, spect=False):
        super().__init__()
        self.spect = spect
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
        self.dropout = nn.Dropout(p=dropout)
        self.pwconv2 = nn.Linear(int(expan_ratio * dim), dim)
        self.se = SE_Module(channel=dim, reduction=16)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv2(self.dwconv1(x) + input)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv2(self.dropout(self.act(self.pwconv1(self.norm(x)))))
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = self.se(input + self.drop_path(x))
        if self.spect:
            return x
        return x + input

    
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
    def __init__(self, in_channels, out_channels, lfae_depth, lfae_kernel_size, gfae_depth, feat_size, train,
                spect, spect_all, dk_ratio, n_heads, use_flash, avgpool, add_pos, learnable_pos, partbi, fullbi, drop_path=0.0, dropout=0.0):
        super().__init__()
        self.downsample = DownsampleLayer(in_channels, out_channels)
        self.lfae_depth = lfae_depth
        self.gfae_depth = gfae_depth
        self.add_pos = add_pos
        self.learnable_pos = learnable_pos
        self.spect = spect
        self.spect_all = spect_all
        self.partbi = partbi
        self.fullbi = fullbi

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

        if not self.spect and not self.spect_all:
            self.lfae_layers = nn.ModuleList([
                LFAEncoder(dim=out_channels, kernel_s=lfae_kernel_size, train=train, dropout=dropout, drop_path=drop_path)
                for _ in range(lfae_depth)
            ])
        if self.spect_all:
            self.lfae_layers = nn.ModuleList([])
            for _ in range(lfae_depth):
                    self.lfae_layers.append(
                        DualFunc(
                            SpectralGatingNetwork(out_channels, feat_size, feat_size / 2 + 1),
                            LFAEncoder(dim=out_channels, kernel_s=lfae_kernel_size, train=train,
                                       dropout=dropout, drop_path=drop_path, spect=True),
                        )
                    )
        if self.spect:
            self.lfae_layers = nn.ModuleList([SpectralGatingNetwork(out_channels, feat_size, feat_size / 2 + 1)])
            for _ in range(lfae_depth):
                self.lfae_layers.append(LFAEncoder(dim=out_channels, kernel_s=lfae_kernel_size, train=train,
                                                   dropout=dropout, drop_path=drop_path, spect=True)
                )
        
        if gfae_depth > 0:
            self.gfae_layers = nn.ModuleList([
                GFAEncoder(dim=out_channels, dropout=dropout, drop_path=drop_path, dk_ratio=dk_ratio, n_heads=n_heads, use_flash=use_flash,
                           avgpool=avgpool) for _ in range(gfae_depth)
            ])
        else:
            self.gfae_layers = nn.Identity()

        if self.fullbi or self.partbi:
            self.mixer = nn.Conv2d(out_channels, out_channels, 1, 1, 0)
    
    def _trunc_normal_(self, tensor, mean=0., std=1.):
        trunc_normal_(tensor, mean=mean, std=std)

    def forward(self, x):
        if self.gfae_depth > 0:
            x = self.downsample(x)
        
        if self.add_pos:
            lfae_x = self.pos_embed_lfae(x)
        if self.learnable_pos:
            lfae_x = x + self.pos_embed_lfae
        for lfae in self.lfae_layers:
            lfae_x = lfae(lfae_x)
        
        if self.gfae_depth > 0:
            if self.fullbi or self.partbi:
                if self.add_pos:
                    gfae_x = self.pos_embed_gfae(x)
                if self.learnable_pos:
                    gfae_x = x + self.pos_embed_gfae
            else:
                if self.add_pos:
                    gfae_x = self.pos_embed_gfae(lfae_x)
                if self.learnable_pos:
                    gfae_x = lfae_x + self.pos_embed_gfae
            for gfae in self.gfae_layers:
                gfae_x = gfae(gfae_x)

            if self.fullbi:
                local2global = torch.sigmoid(gfae_x)
                global2local = torch.sigmoid(lfae_x)
                local_feat = lfae_x * local2global
                global_feat = gfae_x * global2local
                return self.mixer(local_feat * global_feat)
            elif self.partbi:
                local2global = torch.sigmoid(gfae_x)
                local_feat = lfae_x * local2global
                return self.mixer(local_feat * gfae_x)
            else:
                return gfae_x
        
        return lfae_x

class SEBlock(nn.Module):
    """ Squeeze and Excite module.

        Pytorch implementation of `Squeeze-and-Excitation Networks` -
        https://arxiv.org/pdf/1709.01507.pdf
    """

    def __init__(self,
                 in_channels: int,
                 rd_ratio: float = 0.0625) -> None:
        """ Construct a Squeeze and Excite Module.

        :param in_channels: Number of input channels.
        :param rd_ratio: Input channel reduction ratio.
        """
        super(SEBlock, self).__init__()
        self.reduce = nn.Conv2d(in_channels=in_channels,
                                out_channels=int(in_channels * rd_ratio),
                                kernel_size=1,
                                stride=1,
                                bias=True)
        self.expand = nn.Conv2d(in_channels=int(in_channels * rd_ratio),
                                out_channels=in_channels,
                                kernel_size=1,
                                stride=1,
                                bias=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """ Apply forward pass. """
        b, c, h, w = inputs.size()
        x = F.avg_pool2d(inputs, kernel_size=[h, w])
        x = self.reduce(x)
        x = F.relu(x)
        x = self.expand(x)
        x = torch.sigmoid(x)
        x = x.view(-1, c, 1, 1)
        return inputs * x


class MobileOneBlock(nn.Module):
    """ MobileOne building block.

        This block has a multi-branched architecture at train-time
        and plain-CNN style architecture at inference time
        For more details, please refer to our paper:
        `An Improved One millisecond Mobile Backbone` -
        https://arxiv.org/pdf/2206.04040.pdf
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 groups: int = 1,
                 inference_mode: bool = False,
                 use_se: bool = False,
                 num_conv_branches: int = 1) -> None:
        """ Construct a MobileOneBlock module.

        :param in_channels: Number of channels in the input.
        :param out_channels: Number of channels produced by the block.
        :param kernel_size: Size of the convolution kernel.
        :param stride: Stride size.
        :param padding: Zero-padding size.
        :param dilation: Kernel dilation factor.
        :param groups: Group number.
        :param inference_mode: If True, instantiates model in inference mode.
        :param use_se: Whether to use SE-ReLU activations.
        :param num_conv_branches: Number of linear conv branches.
        """
        super(MobileOneBlock, self).__init__()
        self.inference_mode = inference_mode
        self.groups = groups
        self.stride = stride
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_conv_branches = num_conv_branches

        # Check if SE-ReLU is requested
        if use_se:
            self.se = SEBlock(out_channels)
        else:
            self.se = nn.Identity()
        self.activation = nn.ReLU()

        if inference_mode:
            self.reparam_conv = nn.Conv2d(in_channels=in_channels,
                                          out_channels=out_channels,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=padding,
                                          dilation=dilation,
                                          groups=groups,
                                          bias=True)
        else:
            # Re-parameterizable skip connection
            self.rbr_skip = nn.BatchNorm2d(num_features=in_channels) \
                if out_channels == in_channels and stride == 1 else None

            # Re-parameterizable conv branches
            rbr_conv = list()
            for _ in range(self.num_conv_branches):
                rbr_conv.append(self._conv_bn(kernel_size=kernel_size,
                                              padding=padding))
            self.rbr_conv = nn.ModuleList(rbr_conv)

            # Re-parameterizable scale branch
            self.rbr_scale = None
            if kernel_size > 1:
                self.rbr_scale = self._conv_bn(kernel_size=1,
                                               padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Apply forward pass. """
        # Inference mode forward pass.
        if self.inference_mode:
            return self.activation(self.se(self.reparam_conv(x)))

        # Multi-branched train-time forward pass.
        # Skip branch output
        identity_out = 0
        if self.rbr_skip is not None:
            identity_out = self.rbr_skip(x)

        # Scale branch output
        scale_out = 0
        if self.rbr_scale is not None:
            scale_out = self.rbr_scale(x)

        # Other branches
        out = scale_out + identity_out
        for ix in range(self.num_conv_branches):
            out += self.rbr_conv[ix](x)

        return self.activation(self.se(out))

    def reparameterize(self):
        """ Following works like `RepVGG: Making VGG-style ConvNets Great Again` -
        https://arxiv.org/pdf/2101.03697.pdf. We re-parameterize multi-branched
        architecture used at training time to obtain a plain CNN-like structure
        for inference.
        """
        if self.inference_mode:
            return
        kernel, bias = self._get_kernel_bias()
        self.reparam_conv = nn.Conv2d(in_channels=self.rbr_conv[0].conv.in_channels,
                                      out_channels=self.rbr_conv[0].conv.out_channels,
                                      kernel_size=self.rbr_conv[0].conv.kernel_size,
                                      stride=self.rbr_conv[0].conv.stride,
                                      padding=self.rbr_conv[0].conv.padding,
                                      dilation=self.rbr_conv[0].conv.dilation,
                                      groups=self.rbr_conv[0].conv.groups,
                                      bias=True)
        self.reparam_conv.weight.data = kernel
        self.reparam_conv.bias.data = bias

        # Delete un-used branches
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_conv')
        self.__delattr__('rbr_scale')
        if hasattr(self, 'rbr_skip'):
            self.__delattr__('rbr_skip')

        self.inference_mode = True

    def _get_kernel_bias(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Method to obtain re-parameterized kernel and bias.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L83

        :return: Tuple of (kernel, bias) after fusing branches.
        """
        # get weights and bias of scale branch
        kernel_scale = 0
        bias_scale = 0
        if self.rbr_scale is not None:
            kernel_scale, bias_scale = self._fuse_bn_tensor(self.rbr_scale)
            # Pad scale branch kernel to match conv branch kernel size.
            pad = self.kernel_size // 2
            kernel_scale = torch.nn.functional.pad(kernel_scale,
                                                   [pad, pad, pad, pad])

        # get weights and bias of skip branch
        kernel_identity = 0
        bias_identity = 0
        if self.rbr_skip is not None:
            kernel_identity, bias_identity = self._fuse_bn_tensor(self.rbr_skip)

        # get weights and bias of conv branches
        kernel_conv = 0
        bias_conv = 0
        for ix in range(self.num_conv_branches):
            _kernel, _bias = self._fuse_bn_tensor(self.rbr_conv[ix])
            kernel_conv += _kernel
            bias_conv += _bias

        kernel_final = kernel_conv + kernel_scale + kernel_identity
        bias_final = bias_conv + bias_scale + bias_identity
        return kernel_final, bias_final

    def _fuse_bn_tensor(self, branch) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Method to fuse batchnorm layer with preceeding conv layer.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L95

        :param branch:
        :return: Tuple of (kernel, bias) after fusing batchnorm.
        """
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = torch.zeros((self.in_channels,
                                            input_dim,
                                            self.kernel_size,
                                            self.kernel_size),
                                           dtype=branch.weight.dtype,
                                           device=branch.weight.device)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim,
                                 self.kernel_size // 2,
                                 self.kernel_size // 2] = 1
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def _conv_bn(self,
                 kernel_size: int,
                 padding: int) -> nn.Sequential:
        """ Helper method to construct conv-batchnorm layers.

        :param kernel_size: Size of the convolution kernel.
        :param padding: Zero-padding size.
        :return: Conv-BN module.
        """
        mod_list = nn.Sequential()
        mod_list.add_module('conv', nn.Conv2d(in_channels=self.in_channels,
                                              out_channels=self.out_channels,
                                              kernel_size=kernel_size,
                                              stride=self.stride,
                                              padding=padding,
                                              groups=self.groups,
                                              bias=False))
        mod_list.add_module('bn', nn.BatchNorm2d(num_features=self.out_channels))
        return mod_list

class ModifiedGDC(nn.Module):
    def __init__(self, image_size, kernel_size, in_chs, in_chs2, num_classes, dropout):
        super(ModifiedGDC, self).__init__()
        self.out_chs = (in_chs + in_chs2)  * 2

        self.conv_dw = nn.Conv2d(in_chs, in_chs, kernel_size=image_size, groups=in_chs, bias=False)
        self.conv_dw2 = nn.Conv2d(in_chs2, in_chs2, kernel_size=kernel_size, groups=in_chs2, bias=False)
        total_in_chs = in_chs + in_chs2
        self.bn1 = nn.BatchNorm2d(total_in_chs)
        self.dropout = nn.Dropout(dropout)
        self.conv = nn.Conv2d(total_in_chs, self.out_chs, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(self.out_chs)
        self.linear = nn.Linear(self.out_chs, num_classes) if num_classes else nn.Identity()

    def forward(self, x, y):
        x = self.conv_dw(x)
        y = self.conv_dw2(y)
        x = torch.cat((x, y), dim=1)
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.bn2(x)
        x = self.linear(x)
        return x


class MixMobileOneNet(nn.Module):
    def __init__(self, variant="XXS", param_variant="s0", img_size=256, num_classes=1000, train=True, add_pos=True, learnable_pos=False, use_flash=False,
                 init_weights=False, stem_spect=False, pe_stem=False, lfae_spect=False, lfae_spect_all=False, partbi=False, fullbi=False, mdgc=False,
                 avgpool=True, dk_ratio=0.2, drop_path=0.0, dropout=0.0, inference_mode=False):
        super().__init__()
        # Define configurations for each variant
        configs_mmn = {
            "XXS": {
                "channels": [24, 48, 88, 168],
                "lfae_depth": [2, 1, 5, 1],
                "lfae_kernel_size": [3, 5, 7, 9],
                "gfae_depth": [0, 1, 1, 1],
                "n_heads":[0,4,4,4],
            },
            "XS": {
                "channels": [32, 64, 100, 192],
                "lfae_depth": [3, 2, 8, 2],
                "lfae_kernel_size": [3, 5, 7, 9],
                "gfae_depth": [0, 1, 1, 1],
                "n_heads":[0,4,4,4],
            },
            "S": {
                "channels": [48, 96, 160, 304],
                "lfae_depth": [3, 2, 8, 2],
                "lfae_kernel_size": [3, 5, 7, 9],
                "gfae_depth": [0, 1, 1, 1],
                "n_heads":[0,4,4,4],
            },
        }

        params_mo = {
            "s0": {"width_multipliers": (0.75, 1.0, 1.0, 2.0),
                "num_conv_branches": 4},
            "s1": {"width_multipliers": (1.5, 1.5, 2.0, 2.5)},
            "s2": {"width_multipliers": (1.5, 2.0, 2.5, 4.0)},
            "s3": {"width_multipliers": (2.0, 2.5, 3.0, 4.0)},
            "s4": {"width_multipliers": (3.0, 3.5, 3.5, 4.0),
                "use_se": True},
        }
        config = configs_mmn[variant]
        self.num_features = config["channels"][-1]

        param = params_mo[param_variant]
        self.num_blocks_per_stage: List[int] = [2, 8, 10, 1]
        self.width_multipliers = param["width_multipliers"]
        assert len(self.width_multipliers) == 4
        self.in_planes = min(64, int(64 * self.width_multipliers[0]))
        self.use_se = param.get("use_se", False)
        self.num_conv_branches = param.get("num_conv_branches", 1)
        self.inference_mode = inference_mode
        
        self.num_classes = num_classes

        self.img_size = img_size // 4

        if not pe_stem:
            self.stem = Stem(input_channels=3, output_channels=config["channels"][0], spect=stem_spect, h=img_size, w=img_size / 2 + 1)
        else:
            self.stem = PatchEmbedding(in_channels=3,out_channels=config["channels"][0], spect=stem_spect, h=img_size, w=img_size / 2 + 1)
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
                spect=lfae_spect,
                spect_all=lfae_spect_all,
                dk_ratio=dk_ratio,
                n_heads=config["n_heads"][i],
                add_pos=add_pos,
                learnable_pos=learnable_pos,
                use_flash=use_flash,
                avgpool=avgpool,
                partbi=partbi,
                fullbi=fullbi,
                drop_path=drop_path,
                dropout=dropout,
            )
            self.stages.append(mmb)
            in_channels = out_channels
            self.img_size = (self.img_size  // 2) + (1 if (self.img_size  // 2) % 2 != 0 else 0)

        # Build stages
        self.stage0 = MobileOneBlock(in_channels=3, out_channels=self.in_planes,
                                     kernel_size=3, stride=2, padding=1,
                                     inference_mode=self.inference_mode)
        self.cur_layer_idx = 1
        self.mobileone_stages = nn.ModuleList()
        stage_configs = [
            (int(64 * self.width_multipliers[0]), self.num_blocks_per_stage[0], 0),
            (int(128 * self.width_multipliers[1]), self.num_blocks_per_stage[1], 0),
            (int(256 * self.width_multipliers[2]), self.num_blocks_per_stage[2], int(self.num_blocks_per_stage[2] // 2) if self.use_se else 0),
            (int(512 * self.width_multipliers[3]), self.num_blocks_per_stage[3], self.num_blocks_per_stage[3] if self.use_se else 0)
        ]

        for out_channels, num_blocks, num_se_blocks in stage_configs:
            stage = self._make_stage(out_channels, num_blocks, num_se_blocks)
            self.mobileone_stages.append(stage)

        self.weight_mmn_x = nn.Parameter(torch.zeros(1, in_channels, self.img_size*2, self.img_size*2))
        self.weight_mo_x = nn.Parameter(torch.zeros(1, int(512 * self.width_multipliers[3]), math.ceil(img_size / 32), math.ceil(img_size / 32)))
        init.xavier_uniform_(self.weight_mmn_x)
        init.xavier_uniform_(self.weight_mo_x)

        if not mdgc:
            self.gap = nn.AdaptiveAvgPool2d(1)
            in_channels = in_channels + int(512 * self.width_multipliers[3])
        self.mdgc = mdgc
        if num_classes > 0:
            if not mdgc:
                self.head = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(in_channels, num_classes)
                )
            else:
                self.head =  ModifiedGDC(self.img_size*2, math.ceil(img_size / 32), in_channels, int(512 * self.width_multipliers[3]), num_classes, dropout)
        else:
            if not mdgc:
                self.head = nn.Identity()
            else:
                self.head =  ModifiedGDC(self.img_size*2, math.ceil(img_size / 32), in_channels, int(512 * self.width_multipliers[3]), num_classes, dropout)

        if init_weights:
            self.apply(self._init_weights)

    def _make_stage(self,
                    planes: int,
                    num_blocks: int,
                    num_se_blocks: int) -> nn.Sequential:
        """ Build a stage of MobileOne model.

        :param planes: Number of output channels.
        :param num_blocks: Number of blocks in this stage.
        :param num_se_blocks: Number of SE blocks in this stage.
        :return: A stage of MobileOne model.
        """
        # Get strides for all layers
        strides = [2] + [1]*(num_blocks-1)
        blocks = []
        for ix, stride in enumerate(strides):
            use_se = False
            if num_se_blocks > num_blocks:
                raise ValueError("Number of SE blocks cannot "
                                 "exceed number of layers.")
            if ix >= (num_blocks - num_se_blocks):
                use_se = True

            # Depthwise conv
            blocks.append(MobileOneBlock(in_channels=self.in_planes,
                                         out_channels=self.in_planes,
                                         kernel_size=3,
                                         stride=stride,
                                         padding=1,
                                         groups=self.in_planes,
                                         inference_mode=self.inference_mode,
                                         use_se=use_se,
                                         num_conv_branches=self.num_conv_branches))
            # Pointwise conv
            blocks.append(MobileOneBlock(in_channels=self.in_planes,
                                         out_channels=planes,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0,
                                         groups=1,
                                         inference_mode=self.inference_mode,
                                         use_se=use_se,
                                         num_conv_branches=self.num_conv_branches))
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.Sequential(*blocks)

    def _trunc_normal_(self, tensor, mean=0., std=1.):
        trunc_normal_(tensor, mean=mean, std=std)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear): 
            self._trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            self._trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        mmn_x = self.stem(x)
        mo_x = self.stage0(x)
        for stage in self.stages:
            mmn_x = stage(mmn_x)

        for stage in self.mobileone_stages:
            mo_x = stage(mo_x)

        if not self.mdgc:
            # Compute weighted sum of the outputs
            weighted_mmn_x = self.gap(self.weight_mmn_x * mmn_x)
            weighted_mo_x = self.gap(self.weight_mo_x * mo_x)
            
            # Concatenate the weighted outputs
            combined_x = torch.cat((weighted_mmn_x, weighted_mo_x), dim=1)
            
            return self.head(combined_x)

        return self.head(self.weight_mmn_x * mmn_x, self.weight_mo_x * mo_x)


def reparameterize_model(model: torch.nn.Module) -> nn.Module:
    """ Method returns a model where a multi-branched structure
        used in training is re-parameterized into a single branch
        for inference.

    :param model: MobileOne model in train mode.
    :return: MobileOne model in inference mode.
    """
    # Avoid editing original graph
    model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'reparameterize'):
            module.reparameterize()
    return model
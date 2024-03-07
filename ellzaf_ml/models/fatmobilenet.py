import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import math
from timm.models.layers import DropPath
from typing import List
import sys
import os


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

def get_conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
    if type(kernel_size) is int:
        use_large_impl = kernel_size > 5
    else:
        assert len(kernel_size) == 2 and kernel_size[0] == kernel_size[1]
        use_large_impl = kernel_size[0] > 5
    has_large_impl = 'LARGE_KERNEL_CONV_IMPL' in os.environ
    if has_large_impl and in_channels == out_channels and out_channels == groups and use_large_impl and stride == 1 and padding == kernel_size // 2 and dilation == 1:
        sys.path.append(os.environ['LARGE_KERNEL_CONV_IMPL'])
        from depthwise_conv2d_implicit_gemm import DepthWiseConv2dImplicitGEMM
        return DepthWiseConv2dImplicitGEMM(in_channels, kernel_size, bias=bias)
    else:
        return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                         padding=padding, dilation=dilation, groups=groups, bias=bias)

class FASA(nn.Module):

    def __init__(self, dim: int, kernel_size: int, num_heads: int, window_size: int):
        super().__init__()
        self.q = nn.Conv2d(dim, dim, 1, 1, 0)
        self.kv = nn.Conv2d(dim, dim*2, 1, 1, 0)
        # self.local_mixer = nn.Conv2d(dim, dim, kernel_size, 1, kernel_size//2, groups=dim)
        if use_lfae:
            self.local_mixer = LFAEncoder(dim=dim, kernel_s=kernel_size, train=True, drop_path=0.)
        else:
            self.local_mixer = get_conv2d(dim, dim, kernel_size, 1, kernel_size//2, 1, dim, True)
        self.mixer = nn.Conv2d(dim, dim, 1, 1, 0)
        self.dim_head = dim // num_heads
        # self.pool = nn.AvgPool2d(window_size, window_size, ceil_mode=False)
        self.pool = self.refined_downsample(dim, window_size, 5)
        self.window_size = window_size
        self.scalor = self.dim_head ** -0.5

    def refined_downsample(self, dim, window_size, kernel_size):
        if window_size==1:
            return nn.Identity()
        for i in range(4):
            if 2**i == window_size:
                break
        block = nn.Sequential()
        for num in range(i):
            # block.add_module('conv{}'.format(num), nn.Conv2d(dim, dim, kernel_size, 2, kernel_size//2, groups=dim))
            block.add_module('conv{}'.format(num), get_conv2d(dim, dim, kernel_size, 2, kernel_size//2, 1, dim, True))
            block.add_module('bn{}'.format(num), nn.BatchNorm(dim))
            if num != i-1:
                # block.add_module('gelu{}'.format(num), nn.GELU())
                block.add_module('linear{}'.format(num), nn.Conv2d(dim, dim, 1, 1, 0))
        return block

    def forward(self, x: torch.Tensor):
        '''
        x: (b c h w)
        '''
        b, c, h, w = x.size()
        H = math.ceil(h/self.window_size)
        W = math.ceil(w/self.window_size)
        q_local = self.q(x)
        q = q_local.reshape(b, -1, self.dim_head, h*w).transpose(-1, -2).contiguous() #(b m (h w) d)
        k, v = self.kv(self.pool(x)).reshape(b, 2, -1, self.dim_head, H*W).permute(1, 0, 2, 4, 3).contiguous() #(b m (H W) d)
        attn = torch.softmax(self.scalor * q @ k.transpose(-1, -2), -1)
        global_feat = attn @ v #(b m (h w) d)
        global_feat = global_feat.transpose(-1, -2).reshape(b, c, h, w)
        local_feat = self.local_mixer(q_local)
        local_weight = torch.sigmoid(local_feat)
        local_feat = local_feat * local_weight
        local2global = torch.sigmoid(global_feat)
        global2local = torch.sigmoid(local_feat)
        local_feat = local_feat * local2global
        # global_feat = global_feat * global2local
        return self.mixer(local_feat * global_feat)
    
class ConvFFN(nn.Module):

    def __init__(self, in_channels, hidden_channels, kernel_size, stride, out_channels):
        super().__init__()
        self.stride = stride
        self.fc1 = nn.Conv2d(in_channels, hidden_channels, 1, 1, 0)
        self.act = nn.GELU()
        # self.dwconv = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, stride, kernel_size//2, groups=hidden_channels)
        self.dwconv = get_conv2d(hidden_channels, hidden_channels, kernel_size, stride, kernel_size//2, 1, hidden_channels, True)
        self.bn = nn.BatchNorm(hidden_channels)
        #self.bn2 = nn.BatchNorm(hidden_channels)
        self.fc2 = nn.Conv2d(hidden_channels, out_channels, 1, 1, 0)
    
    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = self.act(x)
        if self.stride == 1:
            x = x + self.dwconv(x)
        else:
            x = self.dwconv(x)
        x = self.bn(x)
        x = self.fc2(x) #(b c h w)
        return x

class FATBlock(nn.Module):

    def __init__(self, dim: int, out_dim: int, kernel_size: int, num_heads: int, window_size: int, 
                 mlp_kernel_size: int, mlp_ratio: float, stride: int, diff_pos: bool = False, drop_path=0.):
        super().__init__()
        self.dim = dim
        # self.cpe = nn.Conv2d(dim, dim, kernel_size, 1, kernel_size//2, groups=dim)
        if not diff_pos:
            self.cpe = get_conv2d(dim, dim, kernel_size, 1, kernel_size//2, 1, dim, True)
        else:
            self.cpe = nn.Parameter(torch.zeros(1, dim, feat_size, feat_size))
        self._trunc_normal_(self.pos_embed_lfae, std=.02)
        self.mlp_ratio = mlp_ratio
        self.norm1 = nn.GroupNorm(1, dim)
        self.attn = FASA(dim, kernel_size, num_heads, window_size)
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.GroupNorm(1, dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if stride == 1:
            self.downsample = nn.Identity()
        else:
            self.downsample = nn.Sequential(
                #nn.Conv2d(dim, dim, mlp_kernel_size, stride, mlp_kernel_size//2, groups=dim),
                get_conv2d(dim, dim, mlp_kernel_size, stride, mlp_kernel_size//2, 1, dim, True),
                nn.BatchNorm(dim),
                nn.Conv2d(dim, out_dim, 1, 1, 0)
            )
        self.ffn = ConvFFN(dim, mlp_hidden_dim, mlp_kernel_size, stride, out_dim)

    def forward(self, x: torch.Tensor):
        x = x + self.cpe(x)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = self.downsample(x) + self.drop_path(self.ffn(self.norm2(x)))
        return x
    
class FATlayer(nn.Module):

    def __init__(self, depth: int, dim: int, out_dim: int, kernel_size: int, num_heads: int, 
                 window_size: int, mlp_kernel_size: int, mlp_ratio: float, drop_paths=[0., 0.],
                 downsample=True, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList(
            [
            FATBlock(dim, dim, kernel_size, num_heads, window_size, mlp_kernel_size,
                  mlp_ratio, 1, drop_paths[i]) for i in range(depth-1)
            ]
        )
        if downsample:
            self.blocks.append(FATBlock(dim, out_dim, kernel_size, num_heads, window_size,
                                     mlp_kernel_size, mlp_ratio, 2, diff_pos, drop_paths[-1]))
        else:
            self.blocks.append(FATBlock(dim, out_dim, kernel_size, num_heads, window_size,
                                     mlp_kernel_size, mlp_ratio, 1, diff_pos, drop_paths[-1]))
    
    def forward(self, x: torch.Tensor):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return x
    
class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2),
                        nn.BatchNorm(out_channels),
                        nn.ReLU()
                    )
    
    def forward(self, x: torch.Tensor):
        return self.conv(x)


class PatchEmbedding(nn.Module):

    def __init__(self, in_channels=3, out_channels=96):
        super().__init__()
        self.conv1 = BasicBlock(in_channels, out_channels//2, 3, 2)
        self.conv2 = BasicBlock(out_channels//2, out_channels, 3, 2)
        self.conv3 = BasicBlock(out_channels, out_channels, 3, 1)
        self.conv4 = BasicBlock(out_channels, out_channels, 3, 1)
        self.conv5 = nn.Conv2d(out_channels, out_channels, 1, 1, 0)
        self.layernorm = nn.GroupNorm(1, out_channels)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return self.layernorm(x)
    
class FAT(nn.Module):
    def __init__(self, in_chans, num_classes, drop_path_rate=0.1, use_checkpoint=False, variant="B0"):
        super().__init__()
        configs = {
            "B0": {
                "in_chans": 3,
                "embed_dims": [32, 80, 160, 256],
                "depths": [2, 2, 6, 2],
                "kernel_sizes": [3, 5, 7, 9],
                "num_heads": [2, 5, 10, 16],
                "window_sizes": [8, 4, 2, 1],
                "mlp_kernel_sizes": [5, 5, 5, 5],
                "mlp_ratios": [4, 4, 4, 4],
                "drop_path_rate": 0.05,
                "use_checkpoint": False,
            },
            "B1": {
                "in_chans": 3,
                "embed_dims": [48, 96, 192, 384],
                "depths": [2, 2, 6, 2],
                "kernel_sizes": [3, 5, 7, 9],
                "num_heads": [3, 6, 12, 24],
                "window_sizes": [8, 4, 2, 1],
                "mlp_kernel_sizes": [5, 5, 5, 5],
                "mlp_ratios": [4, 4, 4, 4],
                "drop_path_rate": 0.1,
                "use_checkpoint": False,
            },
            "B2": {
                "in_chans": 3,
                "embed_dims": [64, 128, 256, 512],
                "depths": [2, 2, 6, 2],
                "kernel_sizes": [3, 5, 7, 9],
                "num_heads": [2, 4, 8, 16],
                "window_sizes": [8, 4, 2, 1],
                "mlp_kernel_sizes": [5, 5, 5, 5],
                "mlp_ratios": [4, 4, 4, 4],
                "drop_path_rate": 0.1,
                "use_checkpoint": False,
            },
            "B3": {
                "in_chans": 3,
                "embed_dims": [64, 128, 256, 512],
                "depths": [4, 4, 16, 4],
                "kernel_sizes": [3, 5, 7, 9],
                "num_heads": [2, 4, 8, 16],
                "window_sizes": [8, 4, 2, 1],
                "mlp_kernel_sizes": [5, 5, 5, 5],
                "mlp_ratios": [4, 4, 4, 4],
                "drop_path_rate": 0.15,
                "use_checkpoint": False,
            },
        }
        config = configs[variant]
        self.num_classes = num_classes
        self.num_layers = len(config["depths"])
        self.mlp_ratios = mlp_ratios
        self.patch_embed = PatchEmbedding(in_chans,config["embed_dims"][0])
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            if i_layer != self.num_layers-1:
                layer = FATlayer(
                    depth=depths[i_layer],
                    dim=embed_dims[i_layer],
                    out_dim=embed_dims[i_layer+1],
                    kernel_size=kernel_sizes[i_layer],
                    num_heads=num_heads[i_layer],
                    window_size=window_sizes[i_layer],
                    mlp_kernel_size=mlp_kernel_sizes[i_layer],
                    mlp_ratio=mlp_ratios[i_layer],
                    drop_paths=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                    downsample=True,
                    use_checkpoint=use_checkpoint
                )
            else:
                layer = FATlayer(
                    depth=depths[i_layer],
                    dim=embed_dims[i_layer],
                    out_dim=embed_dims[i_layer+1],
                    kernel_size=kernel_sizes[i_layer],
                    num_heads=num_heads[i_layer],
                    window_size=window_sizes[i_layer],
                    mlp_kernel_size=mlp_kernel_sizes[i_layer],
                    mlp_ratio=mlp_ratios[i_layer],
                    drop_paths=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                    downsample=False,
                    use_checkpoint=use_checkpoint
                )
            self.layers.append(layer)
        self.norm = nn.GroupNorm(1, embed_dims[-1])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes>0 else nn.Identity()

    def forward_feature(self, x):
        '''
        x: (b 3 h w)
        '''
        x = self.patch_embed(x)
        for layer in self.layers:
            x = layer(x)
        x = self.avgpool(self.norm(x))
        return x.flatten(1)

    def forward(self, x):
        x = self.forward_feature(x)
        return self.head(x)
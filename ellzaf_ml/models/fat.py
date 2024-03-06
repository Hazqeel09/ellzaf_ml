import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import math
from timm.models.layers import DropPath
from typing import List
import sys
import os
from collections import namedtuple
from packaging import version


Config = namedtuple('FlashAttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])

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

class FASA(nn.Module):

    def __init__(self, dim: int, kernel_size: int, num_heads: int, window_size: int, use_flash: bool = False):
        super().__init__()
        self.use_flash = use_flash
        self.flash_attn = Attend(use_flash)

        self.q = nn.Conv2d(dim, dim, 1, 1, 0)
        self.kv = nn.Conv2d(dim, dim*2, 1, 1, 0)
        self.local_mixer = get_conv2d(dim, dim, kernel_size, 1, kernel_size//2, 1, dim, True)
        self.mixer = nn.Conv2d(dim, dim, 1, 1, 0)
        self.dim_head = dim // num_heads
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
            block.add_module('conv{}'.format(num), get_conv2d(dim, dim, kernel_size, 2, kernel_size//2, 1, dim, True))
            block.add_module('bn{}'.format(num), nn.BatchNorm(dim))
            if num != i-1:
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
        if self.use_flash:
            global_feat = self.flash_attn(q, k, v).reshape(b, H * W, c)
        else:
            attn = torch.softmax(self.scalor * q @ k.transpose(-1, -2), -1)
            global_feat = attn @ v #(b m (h w) d)
        global_feat = global_feat.transpose(-1, -2).reshape(b, c, h, w)
        local_feat = self.local_mixer(q_local)
        local_weight = torch.sigmoid(local_feat)
        local_feat = local_feat * local_weight
        local2global = torch.sigmoid(global_feat)
        global2local = torch.sigmoid(local_feat)
        local_feat = local_feat * local2global
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
                 mlp_kernel_size: int, mlp_ratio: float, stride: int, drop_path=0.):
        super().__init__()
        self.dim = dim
        # self.cpe = nn.Conv2d(dim, dim, kernel_size, 1, kernel_size//2, groups=dim)
        self.cpe = get_conv2d(dim, dim, kernel_size, 1, kernel_size//2, 1, dim, True)
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
                                     mlp_kernel_size, mlp_ratio, 2, drop_paths[-1]))
        else:
            self.blocks.append(FATBlock(dim, out_dim, kernel_size, num_heads, window_size,
                                     mlp_kernel_size, mlp_ratio, 1, drop_paths[-1]))
    
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
                    depths[i_layer],
                    embed_dims[i_layer],
                    embed_dims[i_layer+1],
                    kernel_sizes[i_layer],
                    num_heads[i_layer],
                    window_sizes[i_layer],
                    mlp_kernel_sizes[i_layer],
                    mlp_ratios[i_layer],
                    dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                    True,
                    use_checkpoint
                )
            else:
                layer = FATlayer(
                    depths[i_layer],
                    embed_dims[i_layer],
                    embed_dims[i_layer],
                    kernel_sizes[i_layer],
                    num_heads[i_layer],
                    window_sizes[i_layer],
                    mlp_kernel_sizes[i_layer],
                    mlp_ratios[i_layer],
                    dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                    False,
                    use_checkpoint,
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
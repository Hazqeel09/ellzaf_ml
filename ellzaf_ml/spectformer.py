from collections import namedtuple
from packaging import version
import math

import torch
import torch.nn.functional as F
from torch import nn, einsum

from timm.models.layers import DropPath

from einops import rearrange
from einops.layers.torch import Rearrange


# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def posemb_sincos_2d(patches, temperature = 10_000, dtype = torch.float32):
    _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    y, x = torch.meshgrid(torch.arange(h, device = device), torch.arange(w, device = device), indexing = 'ij')
    assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'
    omega = torch.arange(dim // 4, device = device) / (dim // 4 - 1)
    omega = 1. / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :] 
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim = 1)
    return pe.type(dtype)

# classes

class ResidualSpect(nn.Module):
    def __init__(self, fn1, fn2):
        super().__init__()
        self.fn1 = fn1
        self.fn2 = fn2
    def forward(self, x, **kwargs):
        return self.fn2(self.fn1(x, **kwargs), **kwargs) + x
    
class ResidualAtt(nn.Module):
    def __init__(self, fn1, fn2):
        super().__init__()
        self.fn1 = fn1
        self.fn2 = fn2
    def forward(self, x, **kwargs):
        x = self.fn1(x, **kwargs) + x
        return self.fn2(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
    
class SpectralGatingNetwork(nn.Module):
    def __init__(self, dim, h=14, w=8):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02)
        self.w = w
        self.h = h

    def forward(self, x, spatial_size=None):

        B, N, C = x.shape

        if spatial_size is None:
            a = b = int(math.sqrt(N))
        else:
            a, b = spatial_size

        x = x.view(B, a, b, C)

        x = x.to(torch.float32)

        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(a, b), dim=(1, 2), norm='ortho')

        x = x.reshape(B, N, C)

        return x

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0., drop_path = 0.):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.drop_path(self.net(x))

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    
class SpectAtt(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, h, w, att_dropout = 0., ff_dropout = 0., drop_path = 0., alpha = 4):
        super().__init__()
        self.layers = nn.ModuleList([])
        for i in range(depth):
            if i < alpha:
                self.layers.append(
                    ResidualSpect(
                    PreNorm(dim, SpectralGatingNetwork(dim, h = h, w = w)),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout = ff_dropout, drop_path = drop_path))),
                )
            else:
                self.layers.append(
                    ResidualAtt(
                    PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = att_dropout)),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout = ff_dropout, drop_path = drop_path))),
                )
    def forward(self, x):
        for spect_attn in self.layers:
            x = spect_attn(x)
        return x

class SpectFormer(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64, att_dropout = 0., ff_dropout = 0., drop_path = 0., spect_alpha = 4):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        h = image_size // patch_size
        w = h // 2 + 1

        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b h w (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.transformer = SpectAtt(dim, depth, heads, dim_head, mlp_dim, h, w, att_dropout, ff_dropout, drop_path, spect_alpha)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)

        pe = posemb_sincos_2d(x)
        x = rearrange(x, 'b ... d -> b (...) d') + pe

        x = self.transformer(x)

        x = x.mean(dim = 1)

        return self.mlp_head(x)

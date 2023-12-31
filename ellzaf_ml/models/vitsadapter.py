import torch
import torch.nn.functional as F
import torch.nn as nn
import timm
from timm.models.vision_transformer import VisionTransformer

import torch.nn as nn
import torch.nn.functional as F

import math

class CDC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(CDC, self).__init__() 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 5), stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):

        [C_out,C_in,H_k,W_k] = self.conv.weight.shape
        tensor_zeros = torch.zeros(C_out, C_in, 1, device=self.conv.weight.device)
            
        conv_weight = torch.cat((tensor_zeros, self.conv.weight[:,:,:,0], tensor_zeros, self.conv.weight[:,:,:,1], self.conv.weight[:,:,:,2], self.conv.weight[:,:,:,3], tensor_zeros, self.conv.weight[:,:,:,4], tensor_zeros), 2)
        conv_weight = conv_weight.contiguous().view(C_out, C_in, 3, 3)
        
        out_normal = F.conv2d(input=x, weight=conv_weight, bias=self.conv.bias, stride=self.conv.stride, padding=self.conv.padding)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal 
        else:
            #pdb.set_trace()
            [C_out,C_in, kernel_size,kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)

            return out_normal - self.theta * out_diff
        
class TokenHistogram(nn.Module):
    def __init__(self, C, H, W, NP, dim):
        super(TokenHistogram, self).__init__()
        # Dimensions
        self.C = C
        self.H = H
        self.W = W
        self.NP = NP

        # Convolution layers for mu and gamma
        self.WConv1 = nn.Conv2d(C, C, 1, bias=False)
        self.WConv2 = nn.Conv2d(C, C, 1, bias=False)

        # Linear layer for final projection
        self.linear = nn.Linear(8, dim)

        # Initialize weights and biases
        nn.init.ones_(self.WConv1.weight)
        # nn.init.zeros_(self.WConv1.bias)
        nn.init.zeros_(self.WConv2.weight)
        # nn.init.zeros_(self.WConv2.bias)

    def forward(self, Z_star):
        # Padding and window size
        padding_size = 1
        J = K = 3

        # Calculate mu and gamma using the convolution layers
        mu = self.WConv1(Z_star)
        gamma = self.WConv2(Z_star)

        # Instead of looping, use unfold to extract sliding local blocks
        Z_star_unfolded = F.unfold(Z_star, kernel_size=(3, 3), padding=1)
        mu_unfolded = F.unfold(mu, kernel_size=(3, 3), padding=1)
        gamma_unfolded = F.unfold(gamma, kernel_size=(3, 3), padding=1)

        # Reshape unfolded tensors to [batch_size, C, H, W, J*K]
        Z_star_patches = Z_star_unfolded.view(-1, self.C, 9, self.H, self.W)
        mu_patches = mu_unfolded.view(-1, self.C, 9, self.H, self.W)
        gamma_patches = gamma_unfolded.view(-1, self.C, 9, self.H, self.W)

        # Calculate U for all patches
        U = gamma_patches * (Z_star_patches - mu_patches)

        # Calculate the exponential part of the histogram
        exp_u2 = torch.exp(-U**2)

        # Sum over the J*K (3*3) dimension to compute the histogram
        Z_hist = exp_u2.sum(dim=2)

        # Normalizing the histogram
        Z_hist /= (J * K)

        # Reshape to the expected output dimension
        # reshape to [batch_size, 196, 8]
        Z_hist = Z_hist.reshape(-1, self.NP, 8)
        Y = self.linear(Z_hist)

        return Y


class SAdapter(nn.Module):
    def __init__(self, num_tokens, dim):
        super(SAdapter, self).__init__()
        self.dim = dim
        self.wh = int((math.sqrt(num_tokens)))
        self.num_tokens = num_tokens

        # Assuming the adapter is placed after the multi-head self-attention (MHSA) and
        # before the MLP within each transformer block.
        self.linear1 = nn.Linear(self.dim, 8)
        self.conv1 = nn.Conv2d(8, 8, kernel_size=1)
        self.cdc = CDC(8, 8, kernel_size=1, stride=1, padding=1, dilation=1, groups=1, bias=False, theta=0.7)
        self.histogram = TokenHistogram(8, self.wh, self.wh, self.num_tokens, dim)

    def forward(self, x):
        # Remove the class token to get shape [batch_size, 196, dim]
        class_token, x = x[:, :1, :], x[:, 1:, :]

        # Dimension down using the linear layer
        x = self.linear1(x)  # [batch_size, 196, 8]

        # Reshape to [batch_size, 14, 14, 8]
        x = x.reshape(-1, 14, 14, 8)

        # Permute to [batch_size, 8, 14, 14]
        x = x.permute(0, 3, 1, 2)

        # Apply the convolutional operations
        x_1 = self.conv1(x)
        x_2 = self.cdc(x)

        # Combine the outputs and apply the histogram layer
        x = x_1 + x_2
        x = self.histogram(x)
        x = torch.cat((class_token, x), dim=1)
        return x


class ViTSAdapter(VisionTransformer):
    def __init__(self, model_size="base", *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize the base VisionTransformer model from timm
        if model_size == "tiny":
            self.base_model = timm.create_model('vit_tiny_patch16_224', pretrained=True)
        elif model_size == "small":
            self.base_model = timm.create_model('vit_small_patch16_224', pretrained=True)
        elif model_size == "base":
            self.base_model = timm.create_model('vit_base_patch16_224', pretrained=True)
        elif model_size == "large":
            self.base_model = timm.create_model('vit_large_patch16_224', pretrained=True)
        else:
            raise ValueError(f"Invalid model size '{model_size}'. Must be one of ['tiny', 'small', 'base', 'large'].")
        
        # Initialize the S-Adapter for each block
        self.s_adapters = nn.ModuleList([SAdapter(num_tokens=196, dim=self.embed_dim) for _ in range(len(self.blocks))])

        # Freeze the MHSA, MLP, and Patch Embedding layers
        for param in self.patch_embed.parameters():
            param.requires_grad = False
        for block in self.blocks:
            for param in block.attn.parameters():
                param.requires_grad = False
            for param in block.mlp.parameters():
                param.requires_grad = False

        # Make sure the normalization layers and S-Adapters are trainable
        for block in self.blocks:
            for param in block.norm1.parameters():
                param.requires_grad = True
            for param in block.norm2.parameters():
                param.requires_grad = True
        for s_adapter in self.s_adapters:
            for param in s_adapter.parameters():
                param.requires_grad = True

    def forward_features(self, x):
        x = self.patch_embed(x)  # Patch embedding
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # Class token
        x = torch.cat((cls_token, x), dim=1)  # Concatenate class token
        x = x + self.pos_embed  # Add positional embedding
        x = self.pos_drop(x)  # Positional dropout

        for i, blk in enumerate(self.blocks):
            # Apply the S-Adapter after the first normalization and MHSA
            x_normed = blk.norm1(x)  # Normalization before MHSA
            x_mhsa = blk.attn(x_normed)  # MHSA with residual connection
            x = x_mhsa + self.s_adapters[i](x_normed) + x  # S-Adapter after MHSA

            # Apply the S-Adapter after the MLP and before its normalization
            x_normed = blk.norm2(x)  # Normalization before MLP
            x_mlp = blk.mlp(x_normed)  # MLP with residual connection
            x = x_mlp + self.s_adapters[i](x) + x  # S-Adapter after MLP

        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


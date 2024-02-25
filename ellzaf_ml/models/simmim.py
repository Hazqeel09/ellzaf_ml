import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from .vitspectral import ViTSpectral
from .vitspectralrope import ViTSpectralRoPE
from .mixmobilenet import MixMobileNet


class ViTSpectralForSimMIM(ViTSpectral):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert self.num_classes == 0

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self._trunc_normal_(self.mask_token, std=0.02)

    def _trunc_normal_(self, tensor, mean=0.0, std=1.0):
        trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)

    def forward(self, x, mask):
        x = self.patch_embed(x)

        assert mask is not None
        B, L, _ = x.shape

        mask_token = self.mask_token.expand(B, L, -1)
        w = mask.flatten(1).unsqueeze(-1).type_as(mask_token)
        x = x * (1 - w) + mask_token * w

        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)

        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            x = blk(x, rel_pos_bias=rel_pos_bias)
        x = self.norm(x)

        x = x[:, 1:]
        B, L, C = x.shape
        H = W = int(L**0.5)
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        return x


class ViTSpectralRoPEForSimMIM(ViTSpectralRoPE):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert self.num_classes == 0

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self._trunc_normal_(self.mask_token, std=0.02)

    def _trunc_normal_(self, tensor, mean=0.0, std=1.0):
        trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)

    def forward(self, x, mask):
        x = self.patch_embed(x)

        assert mask is not None
        B, L, _ = x.shape

        mask_token = self.mask_token.expand(B, L, -1)
        w = mask.flatten(1).unsqueeze(-1).type_as(mask_token)
        x = x * (1 - w) + mask_token * w

        if self.pos_embed is not None:
            x += self.pos_embed(x)
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            x = blk(x, rel_pos_bias=rel_pos_bias)
        x = self.norm(x)

        B, L, C = x.shape
        H = W = int(L**0.5)
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        return x
    
class MixMobileNetForSimMIM(MixMobileNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert self.num_classes == 0

        self.mask_token = nn.Parameter(torch.zeros(1, 3, 1, 1))
        self._trunc_normal_(self.mask_token, std=0.02)

    def _trunc_normal_(self, tensor, mean=0.0, std=1.0):
        trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)

    def forward(self, x, mask):
        assert mask is not None
        B, C, H, W = x.shape

        # Resize mask to match the input dimensions
        w = F.interpolate(mask.view(B, 1, 14, 14).float(), size=(H, W), mode='nearest')
        print(w.shape)

        mask_token = self.mask_token.expand(B, C, H, W)
        print(mask_token.shape)
        x = x * (1 - w) + mask_token * w

        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return x

class SimMIM(nn.Module):
    def __init__(self, encoder, encoder_stride):
        super().__init__()
        self.encoder = encoder
        self.encoder_stride = encoder_stride

        if hasattr(self.encoder, 'in_chans') and self.encoder.in_chans is not None:
            self.in_chans = self.encoder.in_chans
        else:
            self.in_chans = 3  # Default value

        if hasattr(self.encoder, 'patch_size') and self.encoder.patch_size is not None:
            self.patch_size = self.encoder.patch_size
            self.decoder = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.encoder.num_features,
                    out_channels=self.encoder_stride**2 * 3,
                    kernel_size=1,
                ),
                nn.PixelShuffle(self.encoder_stride),
            )
        else:
            self.patch_size = 16  # Default value
            self.decoder = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.encoder.num_features,
                    out_channels=28**2 * 3,
                    kernel_size=1,
                ),
                nn.PixelShuffle(28),
            )

    def forward(self, x, mask):
        z = self.encoder(x, mask)
        x_rec = self.decoder(z)

        mask = (
            mask.repeat_interleave(self.patch_size, 1)
            .repeat_interleave(self.patch_size, 2)
            .unsqueeze(1)
            .contiguous()
        )
        loss_recon = F.l1_loss(x, x_rec, reduction="none")
        loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.in_chans
        return loss

    @torch.jit.ignore
    def no_weight_decay(self):
        if hasattr(self.encoder, "no_weight_decay"):
            return {"encoder." + i for i in self.encoder.no_weight_decay()}
        return {}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        if hasattr(self.encoder, "no_weight_decay_keywords"):
            return {"encoder." + i for i in self.encoder.no_weight_decay_keywords()}
        return {}

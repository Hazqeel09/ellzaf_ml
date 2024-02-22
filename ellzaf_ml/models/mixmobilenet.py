import torch

# GFAE
class GFAEncoder(nn.Module):
    def __init__(self, dim, drop_path=0.,expan_ratio=4,n_heads=4,qkv_bias=True):
        super().__init__()
        self.norm1= nn.LayerNorm(dim, eps=1e-6)
        self.n_heads=n_heads
        self.temp=nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv=nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj=nn.Linear(dim, dim)
        self.dk_ratio=0.2
        self.dw=nn.Conv2d(dim,dim, kernel_size=3,padding=3//2,groups=dim)
        self.avgpool= nn.AvgPool2d(2,stride=2)
        self.convtranspose2d=nn.ConvTranspose2d(dim,dim,2,stride=2)
        self.norm2=nn.LayerNorm(dim,eps=1e-6)
        self.pw1=nn.Linear(dim, expan_ratio*dim) # pointwise/1x1 convs
        self.act=nn.GELU()
        self.pw2=nn.Linear(expan_ratio*dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self,x):
        input=x
        x=self.avgpool(self.dw(x)+input)
        B,C,H,W=x.shape
        x=x.reshape(B,C,H*W).permute(0, 2, 1).norm1(x)

        # Attention
        qkv=self.qkv(x).reshape(B,H*W,3,self.n_heads,C//self.n_heads)
        qkv=qkv.permute(2, 0, 3, 1, 4)
        q=torch.nn.functional.normalize(qkv[0].transpose(-2, -1),dim=-1)
        k=torch.nn.functional.normalize(qkv[1].transpose(-2, -1),dim=-1)
        v=qkv[2].transpose(-2, -1)

        # DropKey
        attn=(q@k.transpose(-2,-1))*self.temp
        m_c=torch.ones_like(attn)*self.dk_ratio
        attn=attn+torch.bernoulli(m_c)*-1e12
        attn=attn.softmax(dim=-1)
        x=self.proj((attn@v).permute(0,3,1,2).reshape(B,H*W,C))
        x=(x+self.drop_path(x)).permute(0,2,1).reshape(B,C,H,W)

        #convtranspose
        x=self.convtranspose2d(x).permute(0,2,3,1)

        # MLP
        x=self.pw2(self.act(self.pw1(self.norm2(x)))).permute(0,3,1,2)
        x=input + self.drop_path(x)
        return x

# LFAE
class LFAEncoder(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6,
        expan_ratio=4.0, kernel_s=7):
        super().__init__()
        self.dwconv1=nn.Conv2d(dim,dim,kernel_size=3,padding=3//2,groups=dim)
        # Pconv : https://arxiv.org/abs/2303.03667
        self.dwconv2=Pconv(dim=dim,n_div=4,forward=’split_cat’,kernel_size=kernel_s)
        self.norm=nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1=nn.Linear(dim, int(expan_ratio * dim))
        self.act=nn.GELU()
        self.pwconv2=nn.Linear(int(expan_ratio * dim), dim)
        # SE_Module : https://arxiv.org/abs/1709.01507
        self.se=SE_Module(channel=dim,ratio=16)
        self.drop_path=DropPath(drop_path) if drop_path > 0. else nn.Identity()
        Electronics 2024, 13, 519 18 of 21

    def forward(self, x):
        input = x
        x=self.dwconv2(self.dwconv1(x)+input)
        x=x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x=self.pwconv2(self.act(self.pwconv1(self.norm(x))))
        x=x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        x=self.se(input + self.drop_path(x))
        return x

class DownsampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
    
    def forward(self, x):
        return self.conv(x)

def make_stage(in_channels, out_channels, stage_index, lfae_depth, gfae_depth, drop_path=0.):
    layers = [DownsampleLayer(in_channels, out_channels)]
    for _ in range(lfae_depth):
        layers.append(LFAEncoder(out_channels, drop_path))
    for _ in range(gfae_depth):
        layers.append(GFAEncoder(out_channels, drop_path))
    return nn.Sequential(*layers)

class MixMobileNet(nn.Module):
    def __init__(self, variant='XXS', num_classes=1000):
        super().__init__()
        # Define configurations for each variant
        configs = {
            'XXS': {'channels': [24, 48, 88, 168], 'depths': [2, 2, 6, 2], 'lfae': [2, 1, 5, 1], 'gfae': [0, 1, 1, 1]},
            'XS': {'channels': [32, 64, 100, 192], 'depths': [3, 3, 9, 3], 'lfae': [3, 2, 8, 2], 'gfae': [0, 1, 1, 1]},
            'S': {'channels': [48, 96, 160, 304], 'depths': [3, 3, 9, 3], 'lfae': [3, 2, 8, 2], 'gfae': [0, 1, 1, 1]},
            'M': {'channels': [32, 64, 128, 256], 'depths': [3, 3, 12, 3], 'lfae': [3, 2, 9, 2], 'gfae': [0, 1, 3, 1]},
        }
        config = configs[variant]

        self.stem = Stem(input_channels=3, output_channels=config['channels'][0])
        self.stages = nn.ModuleList()

        # Instantiate stages with the configured number of channels and blocks
        in_channels = config['channels'][0]
        for i, out_channels in enumerate(config['channels'][1:], start=1):
            self.stages.append(make_stage(in_channels, out_channels, stage_index=i, 
                                          lfae_depth=config['lfae'][i-1], gfae_depth=config['gfae'][i-1]))
            in_channels = out_channels

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(config['channels'][-1], num_classes)

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
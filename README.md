# Ellzaf ML
Implementations of ML papers in PyTorch

[![Downloads](https://static.pepy.tech/badge/ellzaf_ml)](https://pepy.tech/project/ellzaf_ml)
[![Downloads](https://static.pepy.tech/badge/ellzaf_ml/month)](https://pepy.tech/project/ellzaf_ml)

## Install
```bash
$ pip install ellzaf_ml
```

## GhostFaceNets
<img src="./images/ghostfacenetsv2.png"></img>
PyTorch version of [GhostFaceNets](https://github.com/HamadYA/GhostFaceNets/tree/main).

GhostNetV2 code from [Huawei Noah's Ark Lab](https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master).

Loss function code from [Insight Face](https://github.com/deepinsight/insightface/blob/master/recognition/arcface_torch/losses.py).

```python
import torch
from ellzaf_ml.ghostfacenetsv2 import ghostfacenetsv2

IMAGE_SIZE = 112

#return embedding
model = ghostfacenetsv2(image_size=IMAGE_SIZE, width=1, dropout=0., args=None)
img = torch.randn(3, 3, IMAGE_SIZE, IMAGE_SIZE)
model(img)

#return classification
model = ghostfacenetsv2(image_size=IMAGE_SIZE, num_classes=3, width=1, dropout=0., args=None)
img = torch.randn(3, 3, IMAGE_SIZE, IMAGE_SIZE)
model(img)
```

In order to not use GAP like mentioned in the paper, you need to specify the image size.

You also need to have image_size>=33.

### TODO
- [x] Replicate model.
- [ ] Create training code.



## SpectFormer
<img src="./images/spectformer.png"></img>

Implementation of [SpectFormer](https://arxiv.org/abs/2304.06446) vanilla architecture.

Code is modified version of ViT from [Vit-PyTorch](https://github.com/lucidrains/vit-pytorch/tree/main).

```python
import torch
from ellzaf_ml.spectformer import SpectFormer

model = SpectFormer(
        image_size = 224,
        patch_size = 16,
        num_classes = 1000,
        dim = 512,
        depth = 12,
        heads = 16,
        mlp_dim = 1024,
        spect_alpha = 4, # amount of spectral block (depth - spect_alpha = attention block)
) 

img = torch.randn(1, 3, 224, 224)
preds =  model(img) # prediction -> (1,1000)
```

SpectFormer utilizes both spectral block and attention block. The amount of spectral block can be speciified using spect_alpha and the remaining block from depth will be attention blocks.

depth - spect_alpha = attention block

12 - 4 = 8

From the code and calculation example above, when spect_alpha are 4 with the depth of 12. The resulting attention block will be 8. If spect_alpha == depth, it will be GFNet while if spect_alpa = 0, it will be ViT.



## LBP and CNN Feature Fusion for face anti-spoofing
<img src="./images/lbpcnnff.png"></img>

Implementation of [LBP and CNN Feature Fusion for face anti-spoofing](https://link.springer.com/article/10.1007/s10044-023-01132-4)

```python
import torch
from ellzaf_ml.lcff import LBPCNNFeatureFusion

model = LBPCNNFeatureFusion(num_classes=2)
img = torch.rand(1, 3, 224, 224)
preds = model(img) # prediction -> (1,2)
```
This model is primarily used for face liveness.

### Experimental
I also modified it to use with other models as backbone after concatenating the features from the two blocks.
You need to specify the number of classes from the backend model instead of LBPCNNFeatureFusion.

You can modify the number of channels after the features are concatenated using adapt and adapt_channels.

In order to obtain the image size for backbone model, you need to divide your current image size with 8.

#### MobileNetV3
We need to use adapt=True so that the number of channels will be 3 instead of 512.
```python
import torch
import timm
from ellzaf_ml.lcff import LBPCNNFeatureFusion

mobilenetv3 = timm.create_model('mobilenetv3_large_100.ra_in1k', pretrained=pretrained)
mobilenetv3.classifier = nn.Linear(self.mobilenetv3.classifier.in_features, 2) #specify number of class here

model = LBPCNNFeatureFusion(backbone="mobilenetv3", adapt=True, backbone_model=mobilenetv3)
img = torch.rand(3, 3, 224, 224)
preds = model(img) # prediction -> (3,2)
```

#### SpectFormer
You can choose to use the 512 channels from the concatenated block output or adapt like MobileNetV3.
```python
import torch
from ellzaf_ml.lcff import LBPCNNFeatureFusion
from ellzaf_ml.spectformer import SpectFormer

spect_m = SpectFormer(
    image_size = 28,
    patch_size = 7,
    num_classes = 2, # specify amount of classes here
    channels = 512, #512 channels if you want to change only the backbone
    dim = 256,
    depth = 12,
    heads = 4,
    mlp_dim = 512,
    att_dropout = 0.01,
    ff_dropout = 0.1,
    spect_alpha = 4, # amount of spectral block (depth - spect_alpha = attention block)
)
model = LBPCNNFeatureFusion(backbone="spectformer", backbone_model=spect_m)
img = torch.rand(3, 3, 224, 224)
preds = model(img) # prediction -> (3,2)
```

#### GhostFaceNets
If you prefer different number of channels instead, you can specify it using adapt_channels.

Note: GhostFaceNets only works with image_size higher than 32.
```python
import torch
from ellzaf_ml.lcff import LBPCNNFeatureFusion
from ellzaf_ml.ghostfacenetsv2 import ghostfacenetsv2

gfn_m = ghostfacenetsv2(image_size=33, width=1,  num_classes=3, channels=10, dropout=0., args=None)

model = LBPCNNFeatureFusion(backbone="ghostfacenets", adapt=True, adapt_channels=10, backbone_model=gfn_m)
img = torch.rand(3, 3, 264, 264)
preds = model(img) # prediction -> (3,2)
```

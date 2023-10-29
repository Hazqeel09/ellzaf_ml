# Ellzaf ML
Implementations of ML papers in PyTorch

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
from ellzaf_ml.ghostfacenetv2 import ghostfacenetv2

IMAGE_SIZE = 112

#return embedding
model = ghostfacenetv2(image_size=IMAGE_SIZE, width=1, dropout=0., args=None)
img = torch.randn(3, 3, IMAGE_SIZE, IMAGE_SIZE)
display(model(img))

#return classification
model = ghostfacenetv2(image_size=IMAGE_SIZE, num_classes=3, width=1, dropout=0., args=None)
img = torch.randn(3, 3, IMAGE_SIZE, IMAGE_SIZE)
model(img)
```

In order to not use GAP like mentioned in the paper, you need to specify the image size.

# TODO
- [x] Replicate model.
- [ ] Create training code.

## SpectFormer
<img src="./images/spectformer.png"></img>

Implementation of [SpectFormer](https://arxiv.org/abs/2304.06446).
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
        dropout = 0.1,
        spect_alpha = 4, # amount of spectral block (depth - spect_alpha = attention block)
        ) 

img = torch.randn(1, 3, 224, 224)
preds =  model(img) #prediction -> (1,1000)
```

SpectFormer utilizes both spectral block and attention block. The amount of spectral block can be speciified using spect_alpha and the remaining block from depth will be attention blocks.

depth - spect_alpha = attention block
12 - 4 = 8

From the code and calculation example above, when spect_alpha are 4 with the depth of 12. The resulting attention block will be 8. If spect_alpha == depth, it will be GFNet while if spect_alpa = 0, it will be ViT.
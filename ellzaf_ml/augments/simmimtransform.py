import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data._utils.collate import default_collate
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


class MaskGenerator:
    def __init__(self, input_size=192, mask_patch_size=32, model_patch_size=4, mask_ratio=0.6):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio
        
        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0
        
        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size
        
        self.token_count = self.rand_size ** 2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))
        
    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1
        
        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)
        
        return mask

def convert_to_rgb(img):
    return img.convert('RGB') if img.mode != 'RGB' else img

def custom_resize_crop(img, target_size=224):
    # Check if the image is smaller than the target size
    if img.size[0] < target_size or img.size[1] < target_size:
        img = T.Resize((target_size, target_size))(img)
    # If the image is larger than the target size, apply center crop
    elif img.size[0] > target_size or img.size[1] > target_size:
        img = T.CenterCrop(target_size)(img)
    return img

class SimMIMTransform:
    def __init__(self, flipped=False):
        transforms_list = [
            T.Lambda(convert_to_rgb),
            T.Lambda(lambda img: custom_resize_crop(img)),
            T.ToTensor(),
            T.Normalize(mean=torch.tensor(IMAGENET_DEFAULT_MEAN), std=torch.tensor(IMAGENET_DEFAULT_STD)),
        ]
        
        # If flipping is enabled, add the flip transform
        if flipped:
            transforms_list.insert(-2, T.RandomHorizontalFlip(p=1))  # Insert before ToTensor
        
        self.transform_img = T.Compose(transforms_list)

        model_patch_size=16
        
        self.mask_generator = MaskGenerator(
            input_size=224,
            mask_patch_size=32,
            model_patch_size=model_patch_size,
            mask_ratio=0.6,
        )
    
    def __call__(self, img):
        img = self.transform_img(img)
        mask = self.mask_generator()
        
        return img, mask


def collate_fn(batch):
    if not isinstance(batch[0][0], tuple):
        return default_collate(batch)
    else:
        batch_num = len(batch)
        ret = []
        for item_idx in range(len(batch[0][0])):
            if batch[0][0][item_idx] is None:
                ret.append(None)
            else:
                ret.append(default_collate([batch[i][0][item_idx] for i in range(batch_num)]))
        ret.append(default_collate([batch[i][1] for i in range(batch_num)]))
        return ret
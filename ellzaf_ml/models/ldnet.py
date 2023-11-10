import torch
import torch.nn as nn
import torch.nn.functional as F

class LDnet(nn.Module):
    def __init__(self, image_size=64, channels=3, dropout=0., activation=nn.ReLU()):
        super(LDnet, self).__init__()

        self.image_wh = int(image_size / 4) * 3 - 2

        self.cnn2d = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=3, padding='same'),
            activation,
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding='same'),
            activation,
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding='same'),
            activation,
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same'),
            activation,
            nn.MaxPool2d(kernel_size=2),
            nn.Upsample(scale_factor=3, mode='bicubic', align_corners=False),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, groups=64), #depthwise conv
            activation,
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1), #pointwise conv
            activation,
        )

        self.cnn3d = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=3, kernel_size=(3,3,3), padding='same'),
            activation,
            nn.Conv3d(in_channels=3, out_channels=1, kernel_size=(3,3,3), padding='same'),
            activation,
        )

        # Fully connected layers
        self.mlp = nn.Sequential(
            nn.Flatten(), # flatten features
            nn.Linear(in_features= 64 * self.image_wh * self.image_wh, out_features=64),
            activation,
            nn.Dropout(p=dropout),
            nn.Linear(in_features=64, out_features=2),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Apply 2D CNN layers
        x = self.cnn2d(x)
        
        # Reshape the 2D CNN output to add the depth dimension before applying 3D CNN layers
        x = x.view(x.size(0), 1, 64, self.image_wh, self.image_wh)  # Reshape to (batch_size, channels, depth, height, width)
        
        x = self.cnn3d(x)
        
        x = self.mlp(x)
        
        return x
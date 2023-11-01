import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class LBPLayer(nn.Module):
    def __init__(self):
        super(LBPLayer, self).__init__()
        # Define the relative positions of neighbors
        self.relative_positions = torch.tensor([
            [-1, -1], [-1, 0], [-1, 1],
            [0, -1],           [0, 1],
            [1, -1], [1, 0], [1, 1]
        ], dtype=torch.long)

    def forward(self, x):
        # Pad the image tensor once, to avoid padding in the loop
        padded_x = F.pad(x, (1, 1, 1, 1), mode='replicate')
        # Convert x to long type to store the LBP codes
        lbp_image = torch.zeros_like(padded_x, dtype=torch.long)
        # Loop through each neighbor position
        for pos in self.relative_positions:
            # Shift the image tensor by the relative position
            shifted_x = padded_x[:, :, 1 + pos[0]:1 + pos[0] + x.shape[2], 1 + pos[1]:1 + pos[1] + x.shape[3]]
            # Update the LBP image with the binary result of the comparison
            lbp_image[:, :, 1:-1, 1:-1] += ((shifted_x >= x).long() * (2 ** (pos[0] * 3 + pos[1] + 4)))
        # Remove the padding
        lbp_image = lbp_image[:, :, 1:-1, 1:-1]
        return lbp_image.float()  # Convert to float for subsequent layers

class LBPCNNFeatureFusion(nn.Module):
    def __init__(self, num_classes=2, adapt=False, adapt_channels=3, backbone=None, backbone_model=None):
        super(LBPCNNFeatureFusion, self).__init__()

        assert backbone is not None and backbone_model is None, 'You need to specify both backbone name and backbone model'
        
        self.backbone = backbone

        self.lbp_layer = LBPLayer()

        self.adapt = adapt
        self.adapt_channels = nn.Conv2d(512, adapt_channels, kernel_size=1)
        
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        if self.backbone == None:
            self.fusion_and_classify = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.AdaptiveAvgPool2d((7, 7)),
                nn.Flatten(),
                nn.Linear(512*7*7, 512),
                nn.ReLU(),
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes)
            )
        else:
            self.backbone_model = backbone_model
            
    def forward(self, x):
        # Convert to grayscale for LBP
        x_gray = x.mean(dim=1, keepdim=True)
        x_lbp = self.lbp_layer(x_gray)

        x_rgb = self.block1(x)
        x_lbp = self.block2(x_lbp)

        cat_x = torch.cat((x_rgb, x_lbp), dim=1)

        if self.adapt:
            cat_x = self.adapt_channels(cat_x)

        if self.backbone:
            x = self.backbone_model(cat_x)
        else:  # If no backbone is specified, use the default series of layers
            x = self.fusion_and_classify(cat_x)
        return x
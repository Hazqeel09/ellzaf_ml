import torch
import torch.nn as nn
import torch.nn.functional as F

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
        # Assuming x is a 4D tensor with shape (batch_size, channels, height, width)
        # Ensure input is a byte tensor
        x = x.byte()
        # Prepare the output tensor
        lbp_image = torch.zeros_like(x, dtype=torch.long)
        # Loop through each neighbor position
        for pos in self.relative_positions:
            # Shift the image tensor by the relative position
            shifted_x = F.pad(x, (1, 1, 1, 1), mode='replicate')
            shifted_x = shifted_x[:, :, pos[0]:pos[0]+x.shape[2], pos[1]:pos[1]+x.shape[3]]
            # Update the LBP image with the binary result of the comparison
            lbp_image += (shifted_x >= x).long()
        return lbp_image

class LBPCNNFeatureFusion(nn.Module):
    def __init__(self):
        super(LBPCNNFeatureFusion, self).__init__()

        self.block1 = nn.ModuleList([
            nn.Conv2d(3, 64, kernel_size=1, padding='same'),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=1, padding='same'),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=1, padding='same'),
            nn.MaxPool2d(2, 2),
        ])

        self.block2 = nn.ModuleList([
            LBPLayer(),
            nn.Conv2d(1, 64, kernel_size=1, padding='same'),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=1, padding='same'),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=1, padding='same'),
            nn.MaxPool2d(2, 2),
        ])

        self.conv4 = nn.Conv2d(512, 512, kernel_size=1, padding='same')
        self.pool = nn.MaxPool2d(2, 2)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=1, padding='same')
        self.pool2 = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.fc = nn.Linear(512 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x_rgb = self.block1(x)
        x_lbp = self.block2(x)

        cat_x = torch.cat((x_rgb, x_lbp), dim=1)
        cat_x = self.conv4(cat_x)
        cat_x = self.pool(cat_x)
        cat_x = self.conv5(cat_x)
        cat_x = self.pool2(cat_x)
        cat_x = self.avgpool(cat_x)

        cat_x = torch.flatten(cat_x, 1)

        cat_x = self.fc(cat_x)
        cat_x = self.fc2(cat_x)
        return self.fc3(cat_x)

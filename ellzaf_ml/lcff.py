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
    def __init__(self, num_classes=2):
        super(LBPCNNFeatureFusion, self).__init__()

        self.lbp_layer = LBPLayer()

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

        self.fusion_conv = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.fc = nn.Linear(512 * 7 * 7, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Convert to grayscale for LBP
        x_gray = x.mean(dim=1, keepdim=True)
        x_lbp = self.lbp_layer(x_gray)

        x_rgb = self.block1(x)
        x_lbp = self.block2(x_lbp)

        cat_x = torch.cat((x_rgb, x_lbp), dim=1)
        cat_x = self.fusion_conv(cat_x)
        cat_x = self.pool(cat_x)
        cat_x = self.avgpool(cat_x)

        cat_x = torch.flatten(cat_x, 1)

        cat_x = self.fc(cat_x)
        cat_x = self.relu(cat_x)
        cat_x = self.fc2(cat_x)
        cat_x = self.relu(cat_x)
        return self.fc3(cat_x)
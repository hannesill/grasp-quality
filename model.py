import torch
import torch.nn as nn

class GQEstimator(nn.Module):
    def __init__(self):
        super(GQEstimator, self).__init__()

        # Input is 1x48x48x48

        # 3D Convolutional Neural Network
        self.conv_block = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1), # 48x48x48x16
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2), # 24x24x24x16
            nn.Conv3d(16, 32, kernel_size=3, padding=1), # 24x24x24x32
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2), # 12x12x12x32
            nn.Conv3d(32, 64, kernel_size=3, padding=1), # 12x12x12x64
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2), # 6x6x6x64
            nn.Conv3d(64, 128, kernel_size=3, padding=1), # 6x6x6x128
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2) # 3x3x3x128
        )

        # Output is 3x3x3x128 -> Flattened 

        # Grasp quality head
        # Input is 3x3x3x128 + 7 (hand pose)
        self.gq_head = nn.Sequential(
            nn.Linear(3 * 3 * 3 * 128 + 7, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, sdf, hand_pose):
        x = self.conv_block(sdf)
        # Flatten the 4d tensor to 1d
        x = x.view(-1, 3 * 3 * 3 * 128)

        # Concatenate hand pose with flattened SDF
        x = torch.cat([x, hand_pose], dim=1)

        grasp_quality = self.gq_head(x)

        return grasp_quality

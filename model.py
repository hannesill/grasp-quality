import torch
import torch.nn as nn

class GQEstimator(nn.Module):
    def __init__(self, input_size=48, base_channels=16, fc_dims=[256, 128, 64]):
        super(GQEstimator, self).__init__()

        print("Initializing GQEstimator")
        print(f"Input size: {input_size}")

        # Input is {input_size}x{input_size}x{input_size}
        
        # Calculate output size after convolutions
        conv_output_size = input_size // 16  # After 4 max pooling layers with stride 2

        # 3D Convolutional Neural Network
        self.conv_block = nn.Sequential(
            nn.Conv3d(1, base_channels, kernel_size=3, padding=1), # 16x48x48x48
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(base_channels, base_channels*2, kernel_size=3, padding=1), # 32x24x24x24
            nn.ReLU(), 
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(base_channels*2, base_channels*4, kernel_size=3, padding=1), # 64x12x12x12
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(base_channels*4, base_channels*8, kernel_size=3, padding=1), # 128x6x6x6
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2) # 128x3x3x3
        )

        # Calculate flattened size
        flattened_size = conv_output_size * conv_output_size * conv_output_size * (base_channels*8)

        print(f"Flattened size: {flattened_size}")

        # Grasp quality head
        layers = []
        prev_dim = flattened_size + 7  # Add 7 for hand pose
        
        for dim in fc_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU()
            ])
            prev_dim = dim
            
        layers.append(nn.Linear(prev_dim, 1))
        self.gq_head = nn.Sequential(*layers)

        # Store for forward pass
        self.flattened_size = flattened_size

        print(f"Number of parameters: {sum(p.numel() for p in self.parameters())}")

    def forward(self, sdf, hand_pose):
        # Add a channel dimension to the SDF
        sdf = sdf.unsqueeze(0) # 1x48x48x48
        
        x = self.conv_block(sdf) # 128x3x3x3

        # Flatten the 4d tensor to 1d
        x = x.view(self.flattened_size) # 128x3x3x3 -> 128*3*3*3 = 3456

        # Concatenate hand pose with flattened SDF
        x = torch.cat([x, hand_pose])

        grasp_quality = self.gq_head(x)

        return grasp_quality

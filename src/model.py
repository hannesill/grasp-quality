import torch
import torch.nn as nn

class ObjectEncoder(nn.Module):
    def __init__(self, input_size=48, base_channels=16):
        super(ObjectEncoder, self).__init__()

        output_size = input_size // 2**4

        print("Initializing ObjectEncoder")
        print(f"Input size: 1 x {input_size} x {input_size} x {input_size}")
        print(f"Output size: {base_channels*8} x {output_size} x {output_size} x {output_size}")

        # 3D Convolutional Neural Network
        self.layers = nn.Sequential(
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

    def forward(self, x):
        return self.layers(x)


class ObjectDecoder(nn.Module):
    def __init__(self, input_size=48, base_channels=16):
        super(ObjectDecoder, self).__init__()

        output_size = input_size // 2**4

        print("Initializing ObjectDecoder")
        print(f"Input size: {base_channels*8} x {output_size} x {output_size} x {output_size}")
        print(f"Output size: 1 x {input_size} x {input_size} x {input_size}")

        # upsample to input size
        self.layers = nn.Sequential(
            nn.Upsample(size=(input_size // 8, input_size // 8, input_size // 8), mode='trilinear'),
            nn.Conv3d(base_channels*8, base_channels*4, kernel_size=3, padding=1), # 64x12x12x12
            nn.ReLU(),
            nn.Upsample(size=(input_size // 4, input_size // 4, input_size // 4), mode='trilinear'),
            nn.Conv3d(base_channels*4, base_channels*2, kernel_size=3, padding=1), # 32x24x24x24
            nn.ReLU(),
            nn.Upsample(size=(input_size // 2, input_size // 2, input_size // 2), mode='trilinear'),
            nn.Conv3d(base_channels*2, base_channels, kernel_size=3, padding=1), # 16x48x48x48
            nn.ReLU(),
            nn.Upsample(size=(input_size, input_size, input_size), mode='trilinear'), # 1x48x48x48
            nn.Conv3d(base_channels, 1, kernel_size=3, padding=1), # 1x48x48x48
        )

    def forward(self, x):
        return self.layers(x)


class ObjectAutoEncoder(nn.Module):
    def __init__(self, input_size=48, base_channels=16):
        super(ObjectAutoEncoder, self).__init__()
        self.encoder = ObjectEncoder(input_size, base_channels)
        self.decoder = ObjectDecoder(input_size, base_channels)
        
    def forward(self, x):
        return self.decoder(self.encoder(x))


class GQEstimator(nn.Module):
    def __init__(self, input_size=48, base_channels=16, fc_dims=[256, 128, 64]):
        super(GQEstimator, self).__init__()

        print("Initializing GQEstimator")

        self.object_encoder = ObjectEncoder(input_size, base_channels)
        
        flattened_size = base_channels * 8 * (input_size // 16) * (input_size // 16) * (input_size // 16)
        
        # Grasp quality head
        layers = []
        prev_size = flattened_size + 19 # DLR Hand II has 7 + 12 DoF (7 for hand pose, 12 for finger pose)
        for dim in fc_dims:
            layers.extend([
                nn.Linear(prev_size, dim),
                nn.ReLU()
            ])
            prev_size = dim
            
        layers.append(nn.Linear(prev_size, 1))
        self.gq_head = nn.Sequential(*layers)

        print(f"Number of parameters: {sum(p.numel() for p in self.parameters())}")

    def encode_sdf(self, sdf):
        """
        Encodes the SDF grid into a flat feature vector.
        """
        # Add channel dimension
        sdf = sdf.unsqueeze(1)
        
        sdf_features = self.object_encoder(sdf)
        return sdf_features.view(sdf_features.size(0), -1)

    def forward(self, combined_features):
        """
        Predicts grasp quality from combined sdf and grasp features.
        """
        pred_quality = self.gq_head(combined_features)
        
        # Reshape: (B, 1) -> (B,)
        return pred_quality.view(-1)

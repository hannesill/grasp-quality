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

    def _make_conv_block(self, in_channels, out_channels, kernel_size=3, stride=1):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )

    def encode_sdf(self, sdf):
        """
        Encodes the SDF into processed feature vector.
        args:
            sdf: (B, 48, 48, 48) or (48, 48, 48) for single sample
        returns: (B, encoded_size) or (encoded_size) for single sample
        """
        # Handle both single samples and batches
        if sdf.dim() == 3:
            # Single sample: (48, 48, 48) -> (1, 1, 48, 48, 48)
            sdf = sdf.unsqueeze(0).unsqueeze(0)
            single_sample = True
        else:
            # Batch: (B, 48, 48, 48) -> (B, 1, 48, 48, 48)
            sdf = sdf.unsqueeze(1)
            single_sample = False

        # Get features through conv stages
        features = self.conv_stages(sdf)
        
        # Global pooling: (B, 64, 3, 3, 3) -> (B, 64, 1, 1, 1)
        features = self.global_pool(features)

        # Flatten: (B, 64, 1, 1, 1) -> (B, 64)
        batch_size = features.shape[0]
        features = features.view(batch_size, -1)
        
        # Encode through spatial encoder
        encoded_features = self.spatial_encoder(features)
        
        # Return appropriate shape
        if single_sample:
            return encoded_features.squeeze(0)  # (encoded_size,)
        else:
            return encoded_features  # (B, encoded_size)

    def forward(self, x):
        """
        Processes a batch of concatenated SDF features and hand poses to predict grasp quality.
        args:
            x: (B, flattened_size + 19) - concatenated SDF features and grasp parameters
        returns: grasp_quality (B,)
        """
        grasp_quality = self.gq_head(x)
        # Reshape: (B, 1) -> (B,)
        grasp_quality = grasp_quality.view(-1)
        return grasp_quality

    def forward_with_sdf(self, sdf_batch, grasp_batch):
        """
        Efficiently processes SDF and grasp data together.
        args:
            sdf_batch: (B, 48, 48, 48) - batch of SDFs
            grasp_batch: (B, 19) - batch of grasp parameters
        returns: grasp_quality (B,)
        """
        # Encode SDFs
        sdf_features = self.encode_sdf(sdf_batch)
        
        # Encode hand pose
        hand_features = self.hand_encoder(grasp_batch)
        
        # Concatenate encoded features and forward pass
        combined_features = torch.cat([sdf_features, hand_features], dim=1)
        grasp_quality = self.gq_head(combined_features)
        
        return grasp_quality.view(-1)
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
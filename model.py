import torch
import torch.nn as nn


class GQEstimator(nn.Module):
    def __init__(self, input_size=48, base_channels=8, spatial_encoder_dims=[32, 32], hand_encoder_dims=[32, 32], gq_head_dims=[64, 32]):
        super(GQEstimator, self).__init__()

        print("Initializing GQEstimator")
        print(f"Input size: {input_size}")

        # 3D Convolutional Neural Network
        self.conv_stages = nn.Sequential(
            # Stage 1: 1x48x48x48 -> 8x24x24x24
            self._make_conv_block(1, base_channels, stride=2),
            # Stage 2: 8x24x24x24 -> 16x12x12x12
            self._make_conv_block(base_channels, base_channels*2, stride=2),
            # Stage 3: 16x12x12x12 -> 32x6x6x6
            self._make_conv_block(base_channels*2, base_channels*4, stride=2),
            # Stage 4: 32x6x6x6 -> 64x3x3x3
            self._make_conv_block(base_channels*4, base_channels*8, stride=2),
        )
        
        # Global pooling: 64x3x3x3 -> 64x1x1x1
        self.global_pool = nn.AdaptiveAvgPool3d(1)

        # Hand encoder
        layers = []
        prev_dim = 19
        for dim in hand_encoder_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
            ])
            prev_dim = dim
        self.hand_encoder = nn.Sequential(*layers)

        # Spatial encoder - processes global pooled features
        layers = []
        prev_dim = base_channels*8  # From global pooling
        for dim in spatial_encoder_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
            ])
            prev_dim = dim
        self.spatial_encoder = nn.Sequential(*layers)

        # Grasp quality head
        layers = []
        prev_dim = spatial_encoder_dims[-1] + hand_encoder_dims[-1]
        for dim in gq_head_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
            ])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 1))
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
        
        # Return appropriate shape
        if single_sample:
            return features.squeeze(0)  # (encoded_size,)
        else:
            return features  # (B, encoded_size)

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

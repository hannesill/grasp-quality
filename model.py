import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttention3D(nn.Module):
    """
    3D Spatial Attention Module that learns to focus on important spatial regions.
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
        
        self.spatial_attention = nn.Sequential(
            nn.Conv3d(channels, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Channel attention
        channel_att = self.channel_attention(x)
        x = x * channel_att
        
        # Spatial attention
        spatial_att = self.spatial_attention(x)
        x = x * spatial_att
        
        return x


class MultiScalePooling3D(nn.Module):
    """
    Multi-scale pooling that preserves spatial information at different scales.
    """
    def __init__(self, channels):
        super().__init__()
        self.pools = nn.ModuleList([
            nn.AdaptiveAvgPool3d(1),      # Global context
            nn.AdaptiveAvgPool3d(2),      # Regional context
            nn.AdaptiveAvgPool3d(3),      # Local context
        ])
        
        # Reduce channels for each scale to prevent explosion
        self.channel_reducers = nn.ModuleList([
            nn.Conv3d(channels, channels // 4, 1),
            nn.Conv3d(channels, channels // 4, 1),
            nn.Conv3d(channels, channels // 2, 1),
        ])
    
    def forward(self, x):
        features = []
        for pool, reducer in zip(self.pools, self.channel_reducers):
            pooled = pool(x)
            reduced = reducer(pooled)
            features.append(reduced.flatten(start_dim=1))
        
        return torch.cat(features, dim=1)


class ImprovedGQEstimator(nn.Module):
    """
    Improved Grasp Quality Estimator with spatial attention and better spatial preservation.
    """
    def __init__(self, input_size=48, base_channels=8, spatial_encoder_dims=[64, 32], 
                 hand_encoder_dims=[32, 32], gq_head_dims=[64, 32], use_attention=True, 
                 use_multiscale=True):
        super().__init__()

        print("Initializing ImprovedGQEstimator")
        print(f"Input size: {input_size}")
        print(f"Using attention: {use_attention}")
        print(f"Using multiscale pooling: {use_multiscale}")

        self.use_attention = use_attention
        self.use_multiscale = use_multiscale

        # 3D Convolutional Neural Network - same as before
        self.conv_stages = nn.Sequential(
            # Stage 1: 1x48x48x48 -> 8x24x24x24
            self._make_conv_block(1, base_channels, stride=1),
            # Stage 2: 8x24x24x24 -> 16x12x12x12
            self._make_conv_block(base_channels, base_channels*2, stride=1),
            # Stage 3: 16x12x12x12 -> 32x6x6x6
            self._make_conv_block(base_channels*2, base_channels*4, stride=1),
            # Stage 4: 32x6x6x6 -> 64x3x3x3
            self._make_conv_block(base_channels*4, base_channels*8, stride=1),
        )
        
        # Add spatial attention before pooling
        if self.use_attention:
            self.spatial_attention = SpatialAttention3D(base_channels*8)
        
        # Choose pooling strategy
        if self.use_multiscale:
            self.pooling = MultiScalePooling3D(base_channels*8)
            # Calculate feature dimensions after multiscale pooling
            # 1×1×1 → channels//4, 2×2×2 → channels//4, 3×3×3 → channels//2
            pooled_dim = (base_channels*8//4) + (base_channels*8//4 * 8) + (base_channels*8//2 * 27)
        else:
            # Alternative: smaller pooling instead of global pooling
            self.pooling = nn.AdaptiveAvgPool3d(2)  # Keep 2×2×2 instead of 1×1×1
            pooled_dim = base_channels*8 * 8  # 64 * 8 = 512
        
        print(f"Pooled feature dimension: {pooled_dim}")

        # Hand encoder
        layers = []
        prev_dim = 19
        for dim in hand_encoder_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            ])
            prev_dim = dim
        self.hand_encoder = nn.Sequential(*layers)

        # Spatial encoder - processes pooled features
        layers = []
        prev_dim = pooled_dim
        for dim in spatial_encoder_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(0.1),
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
                nn.Dropout(0.1),
            ])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 1))
        self.gq_head = nn.Sequential(*layers)

        print(f"Number of parameters: {sum(p.numel() for p in self.parameters())}")

    def _make_conv_block(self, in_channels, out_channels, kernel_size=3, stride=1):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )

    def encode_sdf(self, sdf):
        """
        Encodes the SDF into processed feature vector with spatial attention.
        """
        # Handle both single samples and batches
        if sdf.dim() == 3:
            sdf = sdf.unsqueeze(0).unsqueeze(0)
            single_sample = True
        else:
            sdf = sdf.unsqueeze(1)
            single_sample = False

        # Get features through conv stages
        features = self.conv_stages(sdf)
        
        # Apply spatial attention if enabled
        if self.use_attention:
            features = self.spatial_attention(features)
        
        # Apply pooling strategy
        if self.use_multiscale:
            pooled_features = self.pooling(features)
        else:
            pooled_features = self.pooling(features)
            pooled_features = pooled_features.flatten(start_dim=1)
        
        # Encode through spatial encoder
        encoded_features = self.spatial_encoder(pooled_features)
        
        # Return appropriate shape
        if single_sample:
            return encoded_features.squeeze(0)
        else:
            return encoded_features

    def forward(self, x):
        """
        Processes a batch of concatenated SDF features and hand poses to predict grasp quality.
        """
        grasp_quality = self.gq_head(x)
        return grasp_quality.view(-1)

    def forward_with_sdf(self, sdf_batch, grasp_batch):
        """
        Efficiently processes SDF and grasp data together.
        """
        sdf_features = self.encode_sdf(sdf_batch)
        
        # Encode hand pose
        hand_features = self.hand_encoder(grasp_batch)
        
        # Concatenate encoded features and forward pass
        combined_features = torch.cat([sdf_features, hand_features], dim=1)
        grasp_quality = self.gq_head(combined_features)
        
        return grasp_quality.view(-1)


class EfficientGQEstimator(nn.Module):
    """
    Efficient version that uses depthwise separable convolutions and lighter attention.
    Good balance between spatial preservation and computational cost.
    """
    def __init__(self, input_size=48, base_channels=8, spatial_encoder_dims=[32, 32], 
                 hand_encoder_dims=[32, 32], gq_head_dims=[64, 32]):
        super().__init__()

        print("Initializing EfficientGQEstimator")
        print(f"Input size: {input_size}")

        # Efficient conv blocks with depthwise separable convolutions
        self.conv_stages = nn.Sequential(
            # Stage 1: 1x48x48x48 -> 8x24x24x24
            self._make_efficient_conv_block(1, base_channels, stride=1),
            # Stage 2: 8x24x24x24 -> 16x12x12x12
            self._make_efficient_conv_block(base_channels, base_channels*2, stride=1),
            # Stage 3: 16x12x12x12 -> 32x6x6x6
            self._make_efficient_conv_block(base_channels*2, base_channels*4, stride=1),
            # Stage 4: 32x6x6x6 -> 64x3x3x3
            self._make_efficient_conv_block(base_channels*4, base_channels*8, stride=1),
        )
        
        # Lightweight spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv3d(base_channels*8, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Efficient pooling: keep 2x2x2 spatial resolution
        self.adaptive_pool = nn.AdaptiveAvgPool3d(2)
        pooled_dim = base_channels*8 * 8  # 64 * 8 = 512
        
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

        # Spatial encoder
        layers = []
        prev_dim = pooled_dim
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

    def _make_efficient_conv_block(self, in_channels, out_channels, kernel_size=3, stride=1):
        """Efficient conv block with depthwise separable convolutions."""
        return nn.Sequential(
            # Depthwise convolution
            nn.Conv3d(in_channels, in_channels, kernel_size, stride, padding=1, groups=in_channels),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
            # Pointwise convolution
            nn.Conv3d(in_channels, out_channels, 1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )

    def encode_sdf(self, sdf):
        """Encodes the SDF with efficient spatial attention."""
        # Handle both single samples and batches
        if sdf.dim() == 3:
            sdf = sdf.unsqueeze(0).unsqueeze(0)
            single_sample = True
        else:
            sdf = sdf.unsqueeze(1)
            single_sample = False

        # Get features through conv stages
        features = self.conv_stages(sdf)
        
        # Apply lightweight spatial attention
        attention_weights = self.spatial_attention(features)
        features = features * attention_weights
        
        # Pool to 2x2x2 instead of 1x1x1
        pooled_features = self.adaptive_pool(features)
        pooled_features = pooled_features.flatten(start_dim=1)
        
        # Encode through spatial encoder
        encoded_features = self.spatial_encoder(pooled_features)
        
        # Return appropriate shape
        if single_sample:
            return encoded_features.squeeze(0)
        else:
            return encoded_features

    def forward(self, x):
        """Processes a batch of concatenated SDF features and hand poses."""
        grasp_quality = self.gq_head(x)
        return grasp_quality.view(-1)

    def forward_with_sdf(self, sdf_batch, grasp_batch):
        """Efficiently processes SDF and grasp data together."""
        sdf_features = self.encode_sdf(sdf_batch)
        hand_features = self.hand_encoder(grasp_batch)
        combined_features = torch.cat([sdf_features, hand_features], dim=1)
        grasp_quality = self.gq_head(combined_features)
        return grasp_quality.view(-1)


# Keep the original for backward compatibility
class GQEstimator(nn.Module):
    def __init__(self, input_size=48, base_channels=8, spatial_encoder_dims=[32, 32], hand_encoder_dims=[32, 32], gq_head_dims=[64, 32]):
        super(GQEstimator, self).__init__()

        print("Initializing GQEstimator")
        print(f"Input size: {input_size}")

        # 3D Convolutional Neural Network
        self.conv_stages = nn.Sequential(
            # Stage 1: 1x48x48x48 -> 8x24x24x24
            self._make_conv_block(1, base_channels, stride=1),
            # Stage 2: 8x24x24x24 -> 16x12x12x12
            self._make_conv_block(base_channels, base_channels*2, stride=1),
            # Stage 3: 16x12x12x12 -> 32x6x6x6
            self._make_conv_block(base_channels*2, base_channels*4, stride=1),
            # Stage 4: 32x6x6x6 -> 64x3x3x3
            self._make_conv_block(base_channels*4, base_channels*8, stride=1),
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

import torch
import torch.nn as nn

class SpatialAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv3d(channels, channels//4, 1),
            nn.ReLU(),
            nn.Conv3d(channels//4, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        attention_weights = self.attention(x)  # (B, 1, 6, 6, 6)
        return x * attention_weights  # Element-wise multiplication

class GQEstimator(nn.Module):
    def __init__(self, input_size=48, base_channels=8, fc_dims=[256, 128, 64]):
        super(GQEstimator, self).__init__()

        print("Initializing GQEstimator")
        print(f"Input size: {input_size}")

        # Input is {input_size}x{input_size}x{input_size}
        
        # Calculate output size after convolutions
        conv_output_size = input_size // 8  # After 3 max pooling layers with stride 2

        # 3D Convolutional Neural Network
        self.conv_block = nn.Sequential(
            nn.Conv3d(1, base_channels, kernel_size=3, padding=1), # 8x48x48x48
            nn.BatchNorm3d(base_channels),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(base_channels, base_channels*2, kernel_size=3, padding=1), # 16x24x24x24
            nn.BatchNorm3d(base_channels*2),
            nn.ReLU(), 
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(base_channels*2, base_channels*4, kernel_size=3, padding=1), # 32x12x12x12
            nn.BatchNorm3d(base_channels*4),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2), # 32x6x6x6
        )

        # Add spatial attention
        self.spatial_attention = SpatialAttention(base_channels*4)

        # Calculate flattened size
        flattened_size = conv_output_size * conv_output_size * conv_output_size * (base_channels*4)

        print(f"Flattened size: {flattened_size}")

        # Grasp quality head
        layers = []
        prev_dim = flattened_size + 19  # Add 7 (hand pose) + 12 (fingers)
        
        for dim in fc_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = dim
            
        layers.append(nn.Linear(prev_dim, 1))
        self.gq_head = nn.Sequential(*layers)

        print(f"Number of parameters: {sum(p.numel() for p in self.parameters())}")

    def encode_sdf(self, sdf):
        """
        Encodes the SDF into a flat feature vector.
        args:
            sdf: (B, 48, 48, 48) or (48, 48, 48) for single sample
        returns: (B, flattened_size) or (flattened_size) for single sample
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

        # Get features (B, channel_dim, D, D, D)
        features = self.conv_block(sdf)
        
        # Apply spatial attention
        features = self.spatial_attention(features)

        # Flatten (B, channel_dim, D, D, D) -> (B, flattened_size)
        batch_size = features.shape[0]
        features = features.view(batch_size, -1)
        
        # Return appropriate shape
        if single_sample:
            return features.squeeze(0)  # (flattened_size,)
        else:
            return features  # (B, flattened_size)

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
        
        # Concatenate features and forward pass
        flattened_features = torch.cat([sdf_features, grasp_batch], dim=1)
        grasp_quality = self.gq_head(flattened_features)
        
        return grasp_quality.view(-1)

class GQEstimatorLarge(nn.Module):
    """
    Larger GQ Estimator optimized for A100 GPU utilization.
    
    Key improvements for GPU utilization:
    - More channels and layers
    - Larger fully connected layers
    - More parameters to stress GPU
    - Optimized for A100 throughput
    """
    def __init__(self, input_size=48, base_channels=16, fc_dims=[512, 256, 128]):
        super(GQEstimatorLarge, self).__init__()

        print("Initializing GQEstimatorLarge (A100 Optimized)")
        print(f"Input size: {input_size}")

        # Input is {input_size}x{input_size}x{input_size}
        
        # Calculate output size after convolutions
        conv_output_size = input_size // 8  # After 3 max pooling layers with stride 2

        # Larger 3D Convolutional Neural Network for better GPU utilization
        self.conv_block = nn.Sequential(
            # First block - more channels
            nn.Conv3d(1, base_channels, kernel_size=3, padding=1), # 16x48x48x48
            nn.BatchNorm3d(base_channels),
            nn.ReLU(),
            nn.Conv3d(base_channels, base_channels, kernel_size=3, padding=1), # 16x48x48x48
            nn.BatchNorm3d(base_channels),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2), # 16x24x24x24
            
            # Second block
            nn.Conv3d(base_channels, base_channels*2, kernel_size=3, padding=1), # 32x24x24x24
            nn.BatchNorm3d(base_channels*2),
            nn.ReLU(),
            nn.Conv3d(base_channels*2, base_channels*2, kernel_size=3, padding=1), # 32x24x24x24
            nn.BatchNorm3d(base_channels*2),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2), # 32x12x12x12
            
            # Third block
            nn.Conv3d(base_channels*2, base_channels*4, kernel_size=3, padding=1), # 64x12x12x12
            nn.BatchNorm3d(base_channels*4),
            nn.ReLU(),
            nn.Conv3d(base_channels*4, base_channels*4, kernel_size=3, padding=1), # 64x12x12x12
            nn.BatchNorm3d(base_channels*4),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2) # 64x6x6x6
        )

        # Add spatial attention
        self.spatial_attention = SpatialAttention(base_channels*4)
        
        # Calculate flattened size
        flattened_size = conv_output_size * conv_output_size * conv_output_size * (base_channels*4)

        print(f"Flattened size: {flattened_size}")

        # Much larger grasp quality head for GPU utilization
        layers = []
        prev_dim = flattened_size + 19  # Add 19 grasp parameters
        
        for dim in fc_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(0.1)  # Light dropout for regularization
            ])
            prev_dim = dim
            
        layers.append(nn.Linear(prev_dim, 1))
        self.gq_head = nn.Sequential(*layers)

        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")
        print("This model is optimized for A100 GPU utilization!")

    def encode_sdf(self, sdf):
        """
        Encodes the SDF into a flat feature vector.
        args:
            sdf: (B, 48, 48, 48) or (48, 48, 48) for single sample
        returns: (B, flattened_size) or (flattened_size) for single sample
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

        # Get features (B, channel_dim, D, D, D)
        features = self.conv_block(sdf)
        
        # Apply spatial attention
        features = self.spatial_attention(features)

        # Flatten (B, channel_dim, D, D, D) -> (B, flattened_size)
        batch_size = features.shape[0]
        features = features.view(batch_size, -1)
        
        # Return appropriate shape
        if single_sample:
            return features.squeeze(0)  # (flattened_size,)
        else:
            return features  # (B, flattened_size)

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
        
        # Concatenate features and forward pass
        flattened_features = torch.cat([sdf_features, grasp_batch], dim=1)
        grasp_quality = self.gq_head(flattened_features)
        
        return grasp_quality.view(-1)

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv3d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        attention_weights = self.attention(x)
        return x * attention_weights

class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm3d(out_channels)
            )
    
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class MultiScaleFeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Different kernel sizes for multi-scale features
        self.branch1 = nn.Conv3d(in_channels, out_channels//4, 1)
        self.branch2 = nn.Conv3d(in_channels, out_channels//4, 3, padding=1)
        self.branch3 = nn.Conv3d(in_channels, out_channels//4, 5, padding=2)
        self.branch4 = nn.Sequential(
            nn.MaxPool3d(3, stride=1, padding=1),
            nn.Conv3d(in_channels, out_channels//4, 1)
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        
        out = torch.cat([branch1, branch2, branch3, branch4], dim=1)
        out = self.relu(self.bn(out))
        return out

class GQEstimatorAdvanced(nn.Module):
    """
    Advanced GQ Estimator with multiple architectural improvements:
    
    1. Residual connections for better gradient flow
    2. Multi-scale feature extraction
    3. Channel + spatial attention
    4. Progressive feature fusion
    5. Deeper architecture with skip connections
    6. Better regularization strategies
    """
    def __init__(self, input_size=48, base_channels=32, fc_dims=[512, 256, 128], dropout=0.15):
        super(GQEstimatorAdvanced, self).__init__()

        print("Initializing GQEstimatorAdvanced")
        print(f"Input size: {input_size}")
        print(f"Base channels: {base_channels}")

        # Calculate output sizes after each stage
        self.stage_sizes = [input_size // 2, input_size // 4, input_size // 8, input_size // 16]
        
        # Initial convolution
        self.initial_conv = nn.Sequential(
            nn.Conv3d(1, base_channels, kernel_size=7, padding=3),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # Multi-scale feature extraction stages
        self.stage1 = nn.Sequential(
            MultiScaleFeatureExtractor(base_channels, base_channels),
            ResidualBlock3D(base_channels, base_channels*2, stride=2)
        )
        
        self.stage2 = nn.Sequential(
            MultiScaleFeatureExtractor(base_channels*2, base_channels*2),
            ResidualBlock3D(base_channels*2, base_channels*4, stride=2)
        )
        
        self.stage3 = nn.Sequential(
            MultiScaleFeatureExtractor(base_channels*4, base_channels*4),
            ResidualBlock3D(base_channels*4, base_channels*8, stride=2)
        )
        
        self.stage4 = nn.Sequential(
            MultiScaleFeatureExtractor(base_channels*8, base_channels*8),
            ResidualBlock3D(base_channels*8, base_channels*16, stride=2)
        )
        
        # Attention mechanisms
        self.channel_attention = ChannelAttention(base_channels*16)
        self.spatial_attention = SpatialAttention(base_channels*16)
        
        # Global feature aggregation
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        
        # Calculate flattened size
        conv_output_size = input_size // 16  # After 4 stages of stride-2 reductions
        flattened_size = conv_output_size * conv_output_size * conv_output_size * (base_channels*16)
        global_features = base_channels*16  # From global pooling
        
        print(f"Spatial features size: {flattened_size}")
        print(f"Global features size: {global_features}")
        
        # Fusion layer for spatial + global features
        self.fusion_layer = nn.Sequential(
            nn.Linear(flattened_size + global_features, base_channels*8),
            nn.BatchNorm1d(base_channels*8),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Grasp parameter embedding
        self.grasp_embedding = nn.Sequential(
            nn.Linear(19, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout//2)
        )
        
        # Combined feature processing
        combined_size = base_channels*8 + 64
        
        # Main prediction head with progressive dimensionality reduction
        layers = []
        prev_dim = combined_size
        
        for dim in fc_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = dim
        
        # Final prediction layer
        layers.append(nn.Linear(prev_dim, 1))
        self.gq_head = nn.Sequential(*layers)

        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")

    def encode_sdf(self, sdf):
        """
        Advanced SDF encoding with multi-scale features and attention.
        """
        # Handle both single samples and batches
        if sdf.dim() == 3:
            sdf = sdf.unsqueeze(0).unsqueeze(0)
            single_sample = True
        else:
            sdf = sdf.unsqueeze(1)
            single_sample = False

        # Initial feature extraction
        x = self.initial_conv(sdf)
        
        # Multi-scale feature extraction through stages
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        # Apply attention mechanisms
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        
        # Extract both spatial and global features
        batch_size = x.shape[0]
        spatial_features = x.view(batch_size, -1)
        global_features = self.global_pool(x).view(batch_size, -1)
        
        # Fuse spatial and global features
        combined_features = torch.cat([spatial_features, global_features], dim=1)
        sdf_features = self.fusion_layer(combined_features)
        
        if single_sample:
            return sdf_features.squeeze(0)
        else:
            return sdf_features

    def forward(self, x):
        """
        Process concatenated features to predict grasp quality.
        """
        grasp_quality = self.gq_head(x)
        return grasp_quality.view(-1)

    def forward_with_sdf(self, sdf_batch, grasp_batch):
        """
        Process SDF and grasp data with advanced fusion.
        """
        # Encode SDFs
        sdf_features = self.encode_sdf(sdf_batch)
        
        # Embed grasp parameters
        grasp_features = self.grasp_embedding(grasp_batch)
        
        # Combine features
        combined_features = torch.cat([sdf_features, grasp_features], dim=1)
        
        # Predict grasp quality
        grasp_quality = self.gq_head(combined_features)
        
        return grasp_quality.view(-1)
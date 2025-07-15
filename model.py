import torch
import torch.nn as nn

class SpatialAttention(nn.Module):
    """Spatial attention module for focusing on important spatial regions."""
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
    """
    Optimized Grasp Quality Estimator for 3-channel inputs.
    
    Balanced architecture for computation vs. results:
    - Efficient channel progression: 16 → 32 → 64
    - Maintains spatial resolution: 6×6×6 final feature maps
    - Spatial attention for focusing on contact regions
    - Moderate FC layer sizes for good representation
    
    Input: 3 channels (SDF + palm distance + fingertip distance)
    Output: Grasp quality score
    """
    
    def __init__(self, input_size=48, base_channels=8, fc_dims=[1024, 256, 64]):
        super(GQEstimator, self).__init__()

        print("Initializing Optimized GQEstimator for 3-channel inputs")
        print(f"Input size: {input_size}")
        print(f"Base channels: {base_channels}")
        print(f"FC dimensions: {fc_dims}")

        # Calculate output size after convolutions
        conv_output_size = input_size // 8  # After 3 max pooling layers with stride 2

        # Optimized 3D CNN for 3-channel inputs
        # Progressive channel increase with good spatial preservation
        self.conv_block = nn.Sequential(
            # Block 1: 3 → 16 channels, 48³ → 24³
            nn.Conv3d(3, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            
            # Block 2: 16 → 32 channels, 24³ → 12³
            nn.Conv3d(base_channels, base_channels*2, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_channels*2),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            
            # Block 3: 32 → 64 channels, 12³ → 6³
            nn.Conv3d(base_channels*2, base_channels*4, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_channels*4),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),  # Final: 64×6×6×6
        )

        # Spatial attention on final feature maps
        self.spatial_attention = SpatialAttention(base_channels*4)

        # Calculate flattened size
        flattened_size = conv_output_size * conv_output_size * conv_output_size * (base_channels*4)
        print(f"Flattened size: {flattened_size}")

        # Balanced FC layers for grasp quality prediction
        layers = []
        prev_dim = flattened_size
        
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

        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")
        print("Architecture optimized for 3-channel spatial reasoning!")

    def encode_3channel_input(self, three_channel_input):
        """
        Encodes 3-channel input into features with spatial attention.
        
        Args:
            three_channel_input: (B, 3, 48, 48, 48) or (3, 48, 48, 48)
                - Channel 0: SDF values
                - Channel 1: Distance to palm
                - Channel 2: Distance to closest fingertip
        
        Returns:
            features: (B, flattened_size) or (flattened_size,)
        """
        # Handle both single samples and batches
        if three_channel_input.dim() == 4:
            # Single sample: (3, 48, 48, 48) → (1, 3, 48, 48, 48)
            three_channel_input = three_channel_input.unsqueeze(0)
            single_sample = True
        else:
            # Batch: (B, 3, 48, 48, 48) - already correct shape
            single_sample = False

        # CNN feature extraction
        features = self.conv_block(three_channel_input)  # (B, 64, 6, 6, 6)
        
        # Apply spatial attention
        features = self.spatial_attention(features)  # Focus on important regions
        
        # Flatten for FC layers
        batch_size = features.shape[0]
        features = features.view(batch_size, -1)  # (B, flattened_size)
        
        # Return appropriate shape
        if single_sample:
            return features.squeeze(0)  # (flattened_size,)
        else:
            return features  # (B, flattened_size)

    def forward(self, x):
        """
        Forward pass for pre-flattened features.
        
        Args:
            x: (B, flattened_size) - pre-encoded features
        
        Returns:
            grasp_quality: (B,) - grasp quality scores
        """
        grasp_quality = self.gq_head(x)
        return grasp_quality.view(-1)

    def forward_with_3channel_input(self, three_channel_batch):
        """
        End-to-end forward pass with 3-channel inputs.
        
        Args:
            three_channel_batch: (B, 3, 48, 48, 48) - 3-channel inputs
        
        Returns:
            grasp_quality: (B,) - grasp quality scores
        """
        # Extract features with spatial attention
        features = self.encode_3channel_input(three_channel_batch)
        
        # Predict grasp quality
        grasp_quality = self.gq_head(features)
        
        return grasp_quality.view(-1)


def get_model_info():
    """Get information about the optimized model architecture."""
    return {
        "architecture": "Optimized GQEstimator",
        "input_channels": 3,
        "channel_progression": "16 → 32 → 64",
        "spatial_resolution": "6×6×6 final features",
        "features": [
            "Spatial attention mechanism",
            "Balanced computation vs. accuracy",
            "Efficient for online optimization",
            "Rich spatial understanding"
        ],
        "recommended_config": {
            "base_channels": 16,
            "fc_dims": [512, 256, 128],
            "batch_size": 128,
            "learning_rate": 1e-3
        }
    }


if __name__ == "__main__":
    # Test the optimized model
    print("Testing Optimized GQEstimator...")
    model = GQEstimator()
    
    # Test input
    test_input = torch.randn(2, 3, 48, 48, 48)
    
    # Forward pass
    with torch.no_grad():
        output = model.forward_with_3channel_input(test_input)
    
    print(f"Test input shape: {test_input.shape}")
    print(f"Test output shape: {output.shape}")
    print(f"Output values: {output}")
    
    # Model info
    info = get_model_info()
    print(f"\nModel Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
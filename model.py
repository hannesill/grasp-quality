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
        prev_dim = flattened_size + 19  # Add 7 (hand pose) + 12 (fingers)
        
        for dim in fc_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU()
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
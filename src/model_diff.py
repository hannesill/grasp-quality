import torch
import torch.nn as nn
import math

from .model import ObjectEncoder


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class DiffusionModel(nn.Module):
    def __init__(self, input_size=19, sdf_feature_dim=3456, time_embed_dim=128, hidden_dim=256, n_layers=4, dropout=0.1):
        super().__init__()
        
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.ReLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        self.sdf_proj = nn.Linear(sdf_feature_dim, hidden_dim)

        layers = [
            nn.Linear(input_size + hidden_dim + time_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        ]
        
        for _ in range(n_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            
        layers.append(nn.Linear(hidden_dim, input_size))
        
        self.main = nn.Sequential(*layers)
        
    def forward(self, x, t, sdf_features):
        t_emb = self.time_embed(t)
        sdf_emb = self.sdf_proj(sdf_features)

        x_in = torch.cat([x, sdf_emb, t_emb], dim=-1)
        
        return self.main(x_in)

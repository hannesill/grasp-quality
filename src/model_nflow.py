import torch
from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.permutations import RandomPermutation

def create_nflow_model(input_dim, num_layers=5, hidden_features=64, context_features=None):
    """Creates a normalizing flow model."""
    transforms = []
    for _ in range(num_layers):
        transforms.append(RandomPermutation(features=input_dim))
        transforms.append(MaskedAffineAutoregressiveTransform(
            features=input_dim,
            hidden_features=hidden_features,
            context_features=context_features
        ))
    transform = CompositeTransform(transforms)

    base_distribution = StandardNormal(shape=[input_dim])
    flow = Flow(transform, base_distribution)
    return flow 
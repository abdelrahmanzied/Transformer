import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    """
    Layer Normalization module for transformer architectures.

    Normalizes the input tensor along the feature dimension (last dimension)
    and applies learnable scale (alpha) and shift (bias) parameters.

    Layer normalization helps stabilize and accelerate training by
    normalizing activations within each position across feature dimensions.

    Args:
        eps (float, optional): A small constant added to the denominator for
                              numerical stability. Default: 1e-6

    Attributes:
        alpha (nn.Parameter): (Gamma) Learnable scale parameter (multiplied)
        bias (nn.Parameter): (Beta) Learnable shift parameter (added)
        eps (float): Small constant for numerical stability
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))


    def forward(self, x):
        """
        Applies layer normalization to the input tensor.

        Computes the mean and standard deviation of each position across
        feature dimensions, normalizes the values, and applies learnable
        scale (alpha) and shift (bias) parameters.

        Args:
            x: Input tensor of shape (batch_size, seq_len, features)

        Returns:
            Normalized tensor with same shape as input

        Notes:
            - Mean and standard deviation are calculated along the last dimension
            - keepdim=True preserves dimensions for proper broadcasting
            - A small epsilon value is added to standard deviation for numerical stability
        """
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
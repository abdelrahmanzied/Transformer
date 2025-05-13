import torch
import torch.nn as nn

class FNN(nn.Module):
    """
    Feed Forward Neural Network module for transformer architectures.

    This implements the position-wise feed-forward network described in
    "Attention Is All You Need", consisting of two linear transformations
    with a non-linearity in between.

    Args:
        d_model (int): Input and output dimension (hidden size of the model)
        d_ff (int): Intermediate dimension of the feedforward network
        dropout (float): Dropout probability for regularization

    Attributes:
        linear_1 (nn.Linear): First linear transformation (d_model → d_ff)
        linear_2 (nn.Linear): Second linear transformation (d_ff → d_model)

    Notes:
        - Typically d_ff is larger than d_model (usually 4x larger)
        - Applied identically and independently to each position in the sequence
        - Often referred to as the "position-wise feed-forward network"
    """
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float,
    ):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # W2 and B2

    def forward(self, x):
        """
        Apply the feed-forward network to the input tensor.

        Implements the equation: FFN(x) = max(0, xW₁ + b₁)W₂ + b₂

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Output tensor of the same shape as input (batch_size, seq_len, d_model)

        Notes:
            - Uses ReLU activation between the two linear transformations
            - Applies dropout after the first activation for regularization
            - Maintains input dimensions throughout the transformation
        """
        return self.linear2(self.dropout(torch.relu(self.linear_1(x))))
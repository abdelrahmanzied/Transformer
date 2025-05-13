import math

import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model: int,
        seq_len: int,
        dropout: float
    ):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # (Positional Encoding) Create a matrix of zeros to store positional encodings for each position/dimension combination
        pe = torch.zeros(seq_len, d_model)

        # (position) represent the word inside the sentence --> position indices [0,1,2,...,seq_len-1] as a column vector of shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)

        # Calculate division terms that control frequency scaling in the sinusoidal position embedding formula: 1/(10000^(2i/d_model))
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apple (sin for even) and (cos for odd) positions
        pe[:, 0::2] = torch.sin(position * div_term) # Even indices: 0, 2
        pe[:, 1::2] = torch.cos(position * div_term) # Odd indices: 1, 3

        # Add batch dimension to allow processing multiple sequences at once (1, seq_len, d_model)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Add positional encoding to the input embeddings.

        The positional encoding is added to the input embeddings to provide
        position information since the transformer has no recurrence or convolution.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Tensor with positional encoding added and dropout applied

        Notes:
            - Slices the precomputed positional encodings to match input sequence length:
              - First dimension [:]: Select the single batch dimension
              - Second dimension [:x.shape[1]]: Select positions from 0 to current sequence length
              - Third dimension [:]: Select all embedding features
            - Uses .detach() to prevent gradient flow through positional encodings
            - Applies dropout for regularization
        """
        x = x + self.pe[:, :x.shape[1], :].detach()
        return self.dropout(x)
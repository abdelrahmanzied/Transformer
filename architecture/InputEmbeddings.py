import math

import torch
import torch.nn as nn

class InputEmbeddings(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        """
        :param x: 
        :return: 
        """
        return self.embedding(x) * math.sqrt(self.d_model)
import math

import torch
import torch.nn as nn

class InputEmbeddings(nn.Module):
    def __init__(
        self,
        d_model: int,
        vocab_size: int,
    ):
        super(InputEmbeddings, self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        """
        :param x: 
        :return: 
        """
        return self.embedding(x) * math.sqrt(self.d_model)

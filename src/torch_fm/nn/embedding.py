from typing import List, Sequence

import numpy as np
import torch
import torch.nn as nn


class MultiEmbedding(nn.Module):
    def __init__(self, emb_sizes: Sequence[int], dim: int) -> None:
        """Extended Embedding class for multiple features.

        Args:
            emb_sizes (Sequence[int]): Sequence of the dictionary size of each features.
            dim (int): Embedding dimension for each features.
            device (str, optional): Defaults to 'cpu'.
        """
        super(MultiEmbedding, self).__init__()
        self.embeddings = nn.ModuleList()
        for size in emb_sizes:
            self.embeddings.append(nn.Embedding(size, dim, sparse=True))
        self._init_weights()

    def _init_weights(self) -> None:
        for emb in self.embeddings:
            n = emb.num_embeddings
            nn.init.uniform_(emb.weight, -np.sqrt(1 / n), np.sqrt(1 / n))

    def forward(self, inputs: torch.Tensor) -> List[torch.Tensor]:
        """forward

        Args:
            inputs (torch.Tensor): Input tensor of multiple features.
                [batch_size, len(emb_sizes)].

        Returns:
            List[torch.Tensor]: Embedded tensor list of multiple features.
                [batch_size, len(emb_sizes), dim]
        """
        outputs: List[torch.Tensor] = []

        for idx, emb in enumerate(self.embeddings):
            outputs.append(emb(inputs[:, idx]))

        return outputs

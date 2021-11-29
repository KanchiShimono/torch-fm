from typing import List

import torch
import torch.nn as nn


class DotInteraction(nn.Module):
    def __init__(self, self_interaction: bool = True) -> None:
        super().__init__()
        self.self_interaction = self_interaction
        self.tril_offset = 0 if self.self_interaction else -1

    def forward(
        self, dense_feature: torch.Tensor, sparse_features: List[torch.Tensor]
    ) -> torch.Tensor:
        batch_size, dim = dense_feature.shape

        cat = torch.cat([dense_feature] + sparse_features, dim=1).view(batch_size, -1, dim)
        interaction = torch.bmm(cat, torch.transpose(cat, 1, 2))

        _, ni, nj = interaction.shape
        i, j = torch.tril_indices(ni, nj, offset=self.tril_offset)
        tri_interaction = interaction[:, i, j]

        output: torch.Tensor = torch.cat([dense_feature] + [tri_interaction], dim=1)

        return output


class CatInteraction(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, dense_feature: torch.Tensor, sparse_features: List[torch.Tensor]
    ) -> torch.Tensor:
        output: torch.Tensor = torch.cat([dense_feature] + sparse_features, dim=1)
        return output

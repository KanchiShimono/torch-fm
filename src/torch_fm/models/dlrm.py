from enum import Enum
from typing import List, Optional, Sequence, Union

import torch
import torch.nn as nn

from ..nn import MLP, CatInteraction, DotInteraction, MultiEmbedding


class InteractionType(Enum):
    DOT = 'dot'
    CAT = 'cat'


class DLRM(nn.Module):
    def __init__(
        self,
        dim: int,
        emb_sizes: Sequence[int],
        bottom_mlp_dims: Sequence[int],
        top_mlp_dims: Sequence[int],
        bottom_sigmoid_pos: Optional[int] = None,
        top_sigmoid_pos: Optional[int] = None,
        interaction: Union[str, InteractionType] = InteractionType.DOT,
    ) -> None:
        super(DLRM, self).__init__()
        self.dim = dim
        self.emb_sizes = emb_sizes
        self.bottom_mlp_dims = bottom_mlp_dims
        self.top_mlp_dims = top_mlp_dims
        self.bottom_sigmoid_pos = bottom_sigmoid_pos
        self.top_sigmoid_pos = top_sigmoid_pos
        if isinstance(interaction, str):
            interaction = InteractionType(interaction)

        self.interaction_type = InteractionType(interaction)
        if self.interaction_type == InteractionType.DOT:
            self.interaction = DotInteraction(self_interaction=True)
        else:
            self.interaction = CatInteraction()  # type: ignore[assignment]
        self.multi_emb = MultiEmbedding(self.emb_sizes, self.dim)
        self.bottom_mlp = MLP(self.bottom_mlp_dims, self.bottom_sigmoid_pos)
        self.top_mlp = MLP(self.top_mlp_dims, self.top_sigmoid_pos)

    def forward(self, dense_inputs: torch.Tensor, sparse_inputs: torch.Tensor) -> torch.Tensor:
        dx: torch.Tensor = self.bottom_mlp(dense_inputs)
        sx: List[torch.Tensor] = self.multi_emb(sparse_inputs)
        interaction = self.interaction(dx, sx)
        output: torch.Tensor = self.top_mlp(interaction)

        return output

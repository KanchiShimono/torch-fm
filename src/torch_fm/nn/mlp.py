from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, dims: Sequence[int], sigmoid_pos: Optional[int] = None) -> None:
        super(MLP, self).__init__()
        layers = nn.ModuleList()
        if sigmoid_pos == -1:
            sigmoid_pos = len(dims) - 2

        for i, dim in enumerate(dims[:-1]):
            layers.append(nn.Linear(dim, dims[i + 1], bias=True))
            if i == sigmoid_pos:
                layers.append(nn.Sigmoid())
            else:
                layers.append(nn.ReLU())

        self.layers = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                nn.init.normal_(
                    mod.weight.data, 0.0, np.sqrt(2.0 / (mod.in_features + mod.out_features))
                )
                nn.init.normal_(mod.bias.data, 0.0, np.sqrt(1.0 / mod.out_features))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output: torch.Tensor = self.layers(inputs)
        return output

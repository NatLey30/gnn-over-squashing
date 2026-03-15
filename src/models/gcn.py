import torch
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool


class GCN(torch.nn.Module):
    """
    GCN architecture for graph classification.

    After message passing layers, node embeddings are aggregated
    using global mean pooling to produce a graph-level embedding.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int
    ):
        super().__init__()

        self.convs = torch.nn.ModuleList()

        # first layer
        self.convs.append(GCNConv(in_channels, hidden_channels))

        # hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        # classifier
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor = None
    ) -> torch.Tensor:

        if batch:
            # message passing
            for conv in self.convs:
                x = conv(x, edge_index)
                x = F.relu(x)

            # graph pooling
            x = global_mean_pool(x, batch)

            # final classification
            x = self.lin(x)

            return x

        else:
            for conv in self.convs[:-1]:
                x = conv(x, edge_index)
                x = F.relu(x)

            x = self.convs[-1](x, edge_index)

            return x

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import (
    GCNConv,
    SAGEConv,
    GATConv,
    global_mean_pool,
)


class GCN(torch.nn.Module):
    """
    Graph Convolutional Network for graph classification.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
    ) -> None:
        super().__init__()

        self.convs = torch.nn.ModuleList()

        self.convs.append(GCNConv(in_channels, hidden_channels))

        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
    ) -> Tensor:
        """
        Forward pass producing graph-level logits.

        Parameters
        ----------
        x : Tensor
            Node features
        edge_index : Tensor
            Graph connectivity
        batch : Tensor
            Batch assignment vector

        Returns
        -------
        Tensor
            Graph logits [num_graphs, num_classes]
        """

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)

        x = global_mean_pool(x, batch)

        x = self.lin(x)

        return x


class GraphSAGE(torch.nn.Module):
    """
    GraphSAGE model for graph classification.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
    ) -> None:
        super().__init__()

        self.convs = torch.nn.ModuleList()

        self.convs.append(SAGEConv(in_channels, hidden_channels))

        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))

        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
    ) -> Tensor:

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)

        x = global_mean_pool(x, batch)

        x = self.lin(x)

        return x


class GAT(torch.nn.Module):
    """
    Graph Attention Network for graph classification.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        heads: int = 8,
    ) -> None:
        super().__init__()

        self.convs = torch.nn.ModuleList()

        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads))

        for _ in range(num_layers - 1):
            self.convs.append(
                GATConv(hidden_channels * heads, hidden_channels, heads=heads)
            )

        self.lin = torch.nn.Linear(hidden_channels * heads, out_channels)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
    ) -> Tensor:

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=0.5, training=self.training)

        x = global_mean_pool(x, batch)

        x = self.lin(x)

        return x

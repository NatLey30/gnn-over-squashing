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
        dropout: float,
    ) -> None:
        super().__init__()

        self.convs = torch.nn.ModuleList()

        if in_channels == 1:
            self.embedding = torch.nn.Embedding(28, hidden_channels)
            first_in = hidden_channels
        else:
            self.embedding = None
            first_in = in_channels

        self.convs.append(GCNConv(first_in, hidden_channels))

        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        self.lin = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, out_channels)
        )

        self.dropout = dropout

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
    ) -> Tensor:

        if x.dtype == torch.long:
            x = self.embedding(x.squeeze())

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

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
        dropout: float,
    ) -> None:
        super().__init__()

        self.convs = torch.nn.ModuleList()

        if in_channels == 1:
            self.embedding = torch.nn.Embedding(28, hidden_channels)
            first_in = hidden_channels
        else:
            self.embedding = None
            first_in = in_channels

        self.convs.append(GCNConv(first_in, hidden_channels))

        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))

        self.lin = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, out_channels)
        )

        self.dropout = dropout

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
    ) -> Tensor:

        if x.dtype == torch.long:
            x = self.embedding(x.squeeze())

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

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
        dropout: float,
        heads: int = 8,
    ) -> None:
        super().__init__()

        self.convs = torch.nn.ModuleList()

        if in_channels == 1:
            self.embedding = torch.nn.Embedding(28, hidden_channels)
            first_in = hidden_channels
        else:
            self.embedding = None
            first_in = in_channels

        self.convs.append(GCNConv(first_in, hidden_channels))

        for _ in range(num_layers - 1):
            self.convs.append(
                GATConv(hidden_channels * heads, hidden_channels, heads=heads)
            )

        self.lin = torch.nn.Linear(hidden_channels * heads, out_channels)

        self.dropout = dropout

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
    ) -> Tensor:

        if x.dtype == torch.long:
            x = self.embedding(x.squeeze())

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_mean_pool(x, batch)

        x = self.lin(x)

        return x

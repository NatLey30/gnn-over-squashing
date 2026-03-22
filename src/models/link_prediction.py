import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GCNConv, SAGEConv, GATConv


class GCN(torch.nn.Module):
    """
    Graph Convolutional Network for node classification.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.convs = torch.nn.ModuleList()

        self.convs.append(GCNConv(in_channels, hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        self.dropout = dropout

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : Tensor
            Node feature matrix [num_nodes, num_features]
        edge_index : Tensor
            Graph connectivity matrix

        Returns
        -------
        Tensor
            Node logits [num_nodes, num_classes]
        """

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return x


class GraphSAGE(torch.nn.Module):
    """
    GraphSAGE model for node classification.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.convs = torch.nn.ModuleList()

        self.convs.append(SAGEConv(in_channels, hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))

        self.dropout = dropout

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """
        Forward pass returning node logits.
        """

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GAT(torch.nn.Module):
    """
    Graph Attention Network for node classification.
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

        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads))

        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_channels * heads, hidden_channels, heads=heads)
            )

        self.dropout = dropout

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """
        Forward pass returning node logits.
        """

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return x

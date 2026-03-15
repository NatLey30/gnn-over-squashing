from torch_geometric.datasets import Planetoid, TUDataset
import torch_geometric.transforms as T


def load_cora(root: str = "data"):
    """
    Load the Cora citation network.

    Returns
    -------
    data : Data
        PyG graph object
    num_features : int
    num_classes : int
    """
    dataset = Planetoid(
        root=root,
        name="Cora",
        transform=T.NormalizeFeatures()
    )

    data = dataset[0]

    return data, dataset.num_features, dataset.num_classes


def load_enzymes(root: str = "data"):
    """
    Load the ENZYMES graph classification dataset.

    Unlike Cora, ENZYMES contains many small graphs.

    Returns
    -------
    dataset : TUDataset
    num_features : int
    num_classes : int
    """
    dataset = TUDataset(
        root=root,
        name="ENZYMES"
    )

    return dataset, dataset.num_features, dataset.num_classes
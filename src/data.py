from torch_geometric.datasets import Planetoid, TUDataset, WikipediaNetwork
import torch_geometric.transforms as T


def load_chamaleon(root: str = "data"):
    """
    Load the chamaleon node classification dataset.
    """
    dataset = WikipediaNetwork(
        root=root,
        name="chamaleon",
        transform=T.NormalizeFeatures()
    )

    return dataset, dataset.num_features, dataset.num_classes


def load_cora(root: str = "data"):
    """
    Load the Cora node classification dataset.
    """
    dataset = Planetoid(
        root=root,
        name="Cora",
        transform=T.NormalizeFeatures()
    )

    data = dataset[0]

    return data, dataset.num_features, dataset.num_classes


def load_dd(root: str = "data"):
    """
    Load the DD graph classification dataset.
    """
    dataset = TUDataset(
        root=root,
        name="DD"
    )

    return dataset, dataset.num_features, dataset.num_classes


def load_enzymes(root: str = "data"):
    """
    Load the ENZYMES graph classification dataset.
    """
    dataset = TUDataset(
        root=root,
        name="ENZYMES"
    )

    return dataset, dataset.num_features, dataset.num_classes


def load_mutag(root: str = "data"):
    """
    Load the MUTAG graph classification dataset.
    """
    dataset = TUDataset(
        root=root,
        name="MUTAG"
    )

    return dataset, dataset.num_features, dataset.num_classes


def load_proteins(root: str = "data"):
    """
    Load the PROTEINS graph classification dataset.
    """
    dataset = TUDataset(
        root=root,
        name="PROTEINS"
    )

    return dataset, dataset.num_features, dataset.num_classes

from torch_geometric.datasets import Planetoid, TUDataset, ZINC, QM9
import torch_geometric.transforms as T


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


def load_pubmed(root: str = "data"):
    """
    Load the PubMed node classification dataset.
    """
    dataset = Planetoid(
        root=root,
        name="PubMed",
        transform=T.NormalizeFeatures()
    )

    data = dataset[0]

    return data, dataset.num_features, dataset.num_classes


def load_qm9(root: str = "data"):
    """
    Load the QM9 graph regression dataset.
    """
    path = root + "/qm9"
    dataset = QM9(
        root=path
    )

    return dataset, dataset.num_features


def load_zinc(root: str = "data"):
    """
    Load the Zinc graph regression dataset.
    """
    path = root + "/zinc"
    dataset = ZINC(
        root=path,
        subset=True
    )

    return dataset, dataset.num_features
"""
Experiment pipeline for the ENZYMES graph classification task.

This module handles:
- dataset loading
- train/test split
- batching with DataLoader
- optional graph rewiring
- model training and evaluation

ENZYMES is a graph classification dataset, so models are taken
from `src.models.graph_classification`.
"""

from typing import Tuple, Dict, Any
import torch
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

from src.data import load_qm9, load_zinc
from src.models.graph_regression import GCN, GraphSAGE, GAT
from src.rewiring.virtual_nodes import add_virtual_node
from src.rewiring.ricci_curvature_rewiring import curvature_rewire
from src.training.train import train_graph_regression
from src.training.evaluate import evaluate_graph_regression


def build_model(
    config: Dict[str, Any],
    in_dim: int,
    num_classes: int,
    use_embedding: bool,
    num_atom_types: int,
) -> torch.nn.Module:
    """
    Build a graph classification model.
    """

    hidden = config["hidden_dim"]
    num_layers = config["num_layers"]
    model_name = config["model"].lower()
    dropout = config["dropout"]

    if model_name == "gcn":
        return GCN(in_dim, hidden, num_classes, num_layers, dropout, use_embedding, num_atom_types)

    if model_name == "graphsage":
        return GraphSAGE(in_dim, hidden, num_classes, num_layers, dropout, use_embedding, num_atom_types)

    if model_name == "gat":
        return GAT(in_dim, hidden, num_classes, num_layers, dropout, use_embedding, num_atom_types)

    raise ValueError(f"Unknown model: {model_name}")


def apply_rewiring(
    config: Dict[str, Any],
    dataset,
    logger
):
    """
    Apply graph rewiring to every graph in the dataset.
    """

    rewiring = config.get("rewiring", "none")

    if rewiring == "none":
        logger.info("No rewiring applied")
        return dataset

    if rewiring == "virtual_nodes":
        logger.info("Applying virtual node rewiring")
        return add_virtual_node(dataset)

    if rewiring == "curvature":
        logger.info("Applying Ricci curvature rewiring")
        return curvature_rewire(dataset)

    raise ValueError(f"Unknown rewiring method: {rewiring}")


def run_experiment(
    config: Dict[str, Any],
    logger,
    device: torch.device,
    dataset_name: str,
) -> Tuple[Dict[str, list], Dict[str, float]]:
    """
    Run the full MUTAG experiment.
    """

    logger.info(f"Loading dataset: {dataset_name.upper()}")

    if dataset_name == "qm9":
        dataset, in_dim = load_qm9()
        num_classes = 19
        use_embedding = False

    elif dataset_name == "zinc":
        dataset, in_dim = load_zinc()
        num_classes = 1
        use_embedding = True

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    dataset = apply_rewiring(config, dataset, logger)

    num_graphs = len(dataset)
    train_size = int(0.8 * num_graphs)
    val_size = int(0.1 * num_graphs)
    test_size = num_graphs - train_size - val_size

    split_generator = torch.Generator().manual_seed(config["seed"])

    batch_size = config.get("batch_size", 128)
    num_atom_types = int(dataset.data.x.max()) + 1

    train_dataset, val_dataset, test_dataset = random_split(
                dataset,
                [train_size, val_size, test_size],
                generator=split_generator
            )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config["seed"])
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    model = build_model(config, in_dim, num_classes, use_embedding, num_atom_types).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"]
    )

    logger.info("Starting training")

    history = train_graph_regression(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        epochs=config["epochs"],
        device=device,
    )

    logger.info("Evaluating model")

    test_acc = evaluate_graph_regression(
        model=model,
        loader=test_loader,
        device=device
    )

    metrics = {"test_accuracy": test_acc}

    return history, metrics, model

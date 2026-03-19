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

from src.data import load_collab, load_dd, load_enzymes, load_mutag, load_proteins
from src.models.graph_classification import GCN, GraphSAGE, GAT
from src.rewiring.virtual_nodes import add_virtual_node
from src.rewiring.ricci_curvature_rewiring import curvature_rewire
from src.training.train import train_graph_classification
from src.training.evaluate import evaluate_graph_classification


def build_model(
    config: Dict[str, Any],
    in_dim: int,
    num_classes: int
) -> torch.nn.Module:
    """
    Build a graph classification model.
    """

    hidden = config["hidden_dim"]
    num_layers = config["num_layers"]
    model_name = config["model"].lower()

    if model_name == "gcn":
        return GCN(in_dim, hidden, num_classes, num_layers)

    if model_name == "graphsage":
        return GraphSAGE(in_dim, hidden, num_classes, num_layers)

    if model_name == "gat":
        return GAT(in_dim, hidden, num_classes, num_layers)

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

    if dataset_name == "collab":
        dataset, in_dim, num_classes = load_collab()

    elif dataset_name == "dd":
        dataset, in_dim, num_classes = load_dd()

    elif dataset_name == "enzymes":
        dataset, in_dim, num_classes = load_enzymes()

    elif dataset_name == "mutag":
        dataset, in_dim, num_classes = load_mutag()

    elif dataset_name == "proteins":
        dataset, in_dim, num_classes = load_proteins()

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    dataset = apply_rewiring(config, dataset, logger)

    num_graphs = len(dataset)
    train_size = int(0.8 * num_graphs)
    test_size = num_graphs - train_size

    split_generator = torch.Generator().manual_seed(config["seed"])

    train_dataset, test_dataset = random_split(
        dataset,
        [train_size, test_size],
        generator=split_generator
    )

    batch_size = config.get("batch_size", 32)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config["seed"])
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    model = build_model(config, in_dim, num_classes).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"]
    )

    logger.info("Starting training")

    history = train_graph_classification(
        model=model,
        loader=train_loader,
        optimizer=optimizer,
        device=device,
        epochs=config["epochs"]
    )

    logger.info("Evaluating model")

    test_acc = evaluate_graph_classification(
        model=model,
        loader=test_loader,
        device=device
    )

    metrics = {"test_accuracy": test_acc}

    return history, metrics

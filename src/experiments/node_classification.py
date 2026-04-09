"""
Experiment pipeline for node classification tasks.

This module handles:
- dataset loading
- optional graph rewiring
- model construction
- training
- evaluation

Supported datasets: CORA, PUBMED.
Models are taken from `src.models.node_classification`.
"""
import os

from typing import Tuple, Dict, Any
import torch
from torch import Tensor

from src.data import load_cora, load_pubmed
from src.models.node_classification import GCN, GraphSAGE, GAT
from src.rewiring.virtual_nodes import add_virtual_node
from src.rewiring.ricci_curvature_rewiring import curvature_rewire
from src.training.train import train_node_classification
from src.training.evaluate import evaluate_node_classification


def build_model(
    config: Dict[str, Any],
    in_dim: int,
    num_classes: int
) -> torch.nn.Module:
    """
    Build a node classification model.

    Parameters
    ----------
    config : dict
        Experiment configuration.
    in_dim : int
        Number of node features.
    num_classes : int
        Number of output classes.

    Returns
    -------
    torch.nn.Module
        Instantiated model.
    """

    hidden = config["hidden_dim"]
    num_layers = config["num_layers"]
    model_name = config["model"].lower()
    dropout = config["dropout"]

    if model_name == "gcn":
        return GCN(in_dim, hidden, num_classes, num_layers, dropout)

    if model_name == "graphsage":
        return GraphSAGE(in_dim, hidden, num_classes, num_layers, dropout)

    if model_name == "gat":
        return GAT(in_dim, hidden, num_classes, num_layers, dropout)

    raise ValueError(f"Unknown model: {model_name}")


def apply_rewiring(
    config: Dict[str, Any],
    data,
    logger
):
    """
    Apply optional graph rewiring.

    Parameters
    ----------
    config : dict
        Experiment configuration.
    data : torch_geometric.data.Data
        Graph data.
    logger : Logger
        Logging object.

    Returns
    -------
    Data
        Rewired graph.
    """

    rewiring = config.get("rewiring", "none")

    if rewiring == "none":
        logger.info("No rewiring applied")
        return data

    if rewiring == "virtual_nodes":
        logger.info("Applying virtual node rewiring")
        return add_virtual_node(data, task="node")

    if rewiring == "curvature":
        logger.info("Applying Ricci curvature rewiring")

        data_name = config["dataset"]

        log_path = os.path.join(
            "results",
            "logging",
            f"{data_name}.txt"
        )

        path = os.path.join(
            "data",
            "rewired",
            f"{data_name}.pt"
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)

        if os.path.exists(path):
            logger.info(f"Loading cached rewired dataset from {path}")
            return torch.load(path, weights_only=False)
        
        rewired = curvature_rewire(data, log_path=log_path)

        # guardar
        torch.save(rewired, path)
        logger.info(f"Saved rewired dataset to {path}")

        return rewired

    raise ValueError(f"Unknown rewiring method: {rewiring}")


def run_experiment(
    config: Dict[str, Any],
    logger,
    device: torch.device,
    dataset_name: str,
) -> Tuple[Dict[str, list], Dict[str, float], torch.nn.Module]:
    """
    Run the full experiment.
    """

    logger.info(f"Loading dataset: {dataset_name.upper()}")

    if dataset_name == "cora":
        data, in_dim, num_classes = load_cora()
    
    elif dataset_name == "pubmed":
        data, in_dim, num_classes = load_pubmed()

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    data = apply_rewiring(config, data, logger)
    data = data.to(device)

    model = build_model(config, in_dim, num_classes).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"]
    )

    logger.info("Starting training")

    history = train_node_classification(
        model=model,
        data=data,
        optimizer=optimizer,
        epochs=config["epochs"]
    )

    logger.info("Evaluating model")

    acc = evaluate_node_classification(model, data)

    metrics = {"test_accuracy": acc}

    return history, metrics, model

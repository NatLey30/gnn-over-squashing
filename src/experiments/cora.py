"""
Experiment pipeline for the Cora node classification task.

This module handles:
- dataset loading
- optional graph rewiring
- model construction
- training
- evaluation

Cora is a node classification dataset, so models are taken from
`src.models.node_classification`.
"""

from typing import Tuple, Dict, Any
import torch
from torch import Tensor

from src.data import load_cora
from src.models import node_classification
from src.rewiring.virtual_nodes import add_virtual_node
from src.rewiring.ricci_curvature_rewiring import curvature_rewire
from src.training.train import train_cora
from src.training.evaluate import evaluate_cora


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

    if model_name == "gcn":
        return node_classification.GCN(in_dim, hidden, num_classes, num_layers)

    if model_name == "graphsage":
        return node_classification.GraphSAGE(in_dim, hidden, num_classes, num_layers)

    if model_name == "gat":
        return node_classification.GAT(in_dim, hidden, num_classes, num_layers)

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
        return add_virtual_node(data)

    if rewiring == "curvature":
        logger.info("Applying Ricci curvature rewiring")
        return curvature_rewire(data)

    raise ValueError(f"Unknown rewiring method: {rewiring}")


def run_experiment(
    config: Dict[str, Any],
    logger,
    device: torch.device
) -> Tuple[Dict[str, list], Dict[str, float]]:
    """
    Run the full Cora experiment.

    Parameters
    ----------
    config : dict
        Experiment configuration.
    logger : Logger
        Logger instance.
    device : torch.device
        CPU or CUDA device.

    Returns
    -------
    history : dict
        Training history.
    metrics : dict
        Evaluation metrics.
    """

    logger.info("Loading dataset: Cora")

    data, in_dim, num_classes = load_cora()
    data = apply_rewiring(config, data, logger)
    data = data.to(device)

    model = build_model(config, in_dim, num_classes).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"]
    )

    logger.info("Starting training")

    history = train_cora(
        model=model,
        data=data,
        optimizer=optimizer,
        epochs=config["epochs"]
    )

    logger.info("Evaluating model")

    acc = evaluate_cora(model, data)

    metrics = {"test_accuracy": acc}

    return history, metrics

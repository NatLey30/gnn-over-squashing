"""
Experiment pipeline for graph regression tasks.

This module handles:
- dataset loading
- train/test split
- batching with DataLoader
- optional graph rewiring
- model training and evaluation

Supported datasets: QM9, ZINC.
Models are taken from `src.models.graph_regression`.
"""

import os

import logging
from typing import Tuple, Dict, Any, List, Union

import torch
from torch.utils.data import random_split
from torch_geometric.data import Data, Dataset
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
    out_dim: int,
    use_embedding: bool,
    num_atom_types: int,
) -> torch.nn.Module:
    """Build a graph regression model from the experiment config.

    Args:
        config: Experiment configuration dictionary. Expected keys:
            - hidden_dim  – hidden layer width.
            - num_layers  – number of message-passing layers.
            - model       – one of "gcn", "graphsage", "gat".
            - dropout     – dropout probability.
        in_dim: Dimensionality of the input node features.
        out_dim: Number of regression output dimensions (e.g. 1 for ZINC,
            19 for QM9).
        use_embedding: Whether to use an atom-type embedding layer instead of
            raw node features (required for ZINC).
        num_atom_types: Vocabulary size for the embedding layer. Ignored when
            use_embedding=False.

    Returns:
        An instantiated, untrained graph regression model.

    Raises:
        ValueError: If config["model"] is not a recognised architecture.
    """
    hidden = config["hidden_dim"]
    num_layers = config["num_layers"]
    model_name = config["model"].lower()
    dropout = config["dropout"]

    if model_name == "gcn":
        return GCN(in_dim, hidden, out_dim, num_layers, dropout, use_embedding, num_atom_types)

    if model_name == "graphsage":
        return GraphSAGE(in_dim, hidden, out_dim, num_layers, dropout, use_embedding, num_atom_types)

    if model_name == "gat":
        return GAT(in_dim, hidden, out_dim, num_layers, dropout, use_embedding, num_atom_types)

    raise ValueError(f"Unknown model: {model_name}")


def apply_rewiring(
    config: Dict[str, Any],
    dataset: Union[Dataset, List[Data]],
    logger: logging.Logger,
) -> Union[Dataset, List[Data]]:
    """Apply an optional graph rewiring strategy to every graph in the dataset.

    The rewiring method is read from config["rewiring"].  When set to
    "none" (or absent), the dataset is returned unchanged.

    Args:
        config: Experiment configuration dictionary. Relevant key:
            - rewiring – one of "none", "virtual_nodes",
              "curvature" (default: "none").
        dataset: A PyG Dataset or list of Data objects to rewire.
        logger: Standard-library logger used for progress messages.

    Returns:
        The rewired dataset as a list of Data objects, or the original
        dataset unchanged when no rewiring is requested.

    Raises:
        ValueError: If config["rewiring"] is not a recognised strategy.
    """
    rewiring = config.get("rewiring", "none")

    if rewiring == "none":
        logger.info("No rewiring applied")
        return dataset

    if rewiring == "virtual_nodes":
        logger.info("Applying virtual node rewiring")
        return [add_virtual_node(data, task="regression") for data in dataset]

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
        
        rewired = curvature_rewire(dataset, log_path=log_path)

        # guardar
        torch.save(rewired, path)
        logger.info(f"Saved rewired dataset to {path}")

        return rewired

    raise ValueError(f"Unknown rewiring method: {rewiring}")


def run_experiment(
    config: Dict[str, Any],
    logger: logging.Logger,
    device: torch.device,
    dataset_name: str,
) -> Tuple[Dict[str, list], Dict[str, float], torch.nn.Module]:
    """Run the full graph regression experiment for a given dataset.

    Loads the requested dataset, optionally rewires the graphs, performs an
    80 / 10 / 10 train / val / test split, trains the model, and evaluates it
    on the held-out test set.

    Args:
        config: Experiment configuration dictionary. Expected keys include
            those consumed by :func:`build_model` and :func:`apply_rewiring`,
            plus:
            - seed         – integer random seed for reproducible splits
              and data-loader shuffling.
            - batch_size   – mini-batch size (default: 128).
            - epochs       – number of training epochs.
            - lr           – Adam learning rate.
            - weight_decay – Adam weight-decay coefficient.
        logger: Standard-library logger used for progress messages.
        device: Torch device ("cpu" or "cuda") on which to train.
        dataset_name: One of "qm9", "zinc" (case-insensitive).

    Returns:
        A three-tuple (history, metrics, model) where:
        - history – dictionary of per-epoch training curves (keys depend
          on :func:`train_graph_regression`).
        - metrics – {"test_mae": float} with the final test MAE.
        - model   – the trained :class:`torch.nn.Module`.

    Raises:
        ValueError: If dataset_name is not supported.
    """
    logger.info(f"Loading dataset: {dataset_name.upper()}")

    if dataset_name == "qm9":
        dataset, in_dim = load_qm9()
        out_dim = 19
        use_embedding = False

    elif dataset_name == "zinc":
        dataset, in_dim = load_zinc()
        out_dim = 1
        use_embedding = True

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Compute num_atom_types before rewiring, while dataset is still a PyG
    # Dataset object with a .data accessor. After rewiring it becomes a plain
    # list and that accessor is no longer available.
    num_atom_types: int = int(dataset.data.x.max()) + 1

    dataset = apply_rewiring(config, dataset, logger)

    num_graphs = len(dataset)
    train_size = int(0.8 * num_graphs)
    val_size = int(0.1 * num_graphs)
    test_size = num_graphs - train_size - val_size

    split_generator = torch.Generator().manual_seed(config["seed"])
    batch_size = config.get("batch_size", 128)

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=split_generator,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config["seed"]),
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = build_model(config, in_dim, out_dim, use_embedding, num_atom_types).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
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

    test_mae = evaluate_graph_regression(
        model=model,
        loader=test_loader,
        device=device,
    )

    metrics: Dict[str, float] = {"test_mae": test_mae}

    return history, metrics, model

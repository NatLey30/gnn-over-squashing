import sys
import torch
import json
import os

from src.config import load_config
from src.utils.seed import set_seed
from src.utils.logging import get_logger
from src.utils.plotting import plot_training

from src.data import load_cora, load_enzymes

from src.models.gcn import GCN
from src.models.graphsage import GraphSAGE
from src.models.gat import GAT

from src.rewiring.virtual_nodes import add_virtual_node
from src.rewiring.ricci_curvature_rewiring import curvature_rewire

from src.training.train import train_cora
from src.training.evaluate import evaluate_cora


def load_dataset(config, logger):

    dataset_name = config.get("dataset", "cora").lower()

    if dataset_name == "cora":
        logger.info("Loading dataset: Cora")
        data, in_dim, num_classes = load_cora()
        return dataset_name, data, in_dim, num_classes

    elif dataset_name == "enzymes":
        logger.info("Loading dataset: ENZYMES")
        dataset, in_dim, num_classes = load_enzymes()

        # For simplicity use the first graph
        # (later you may want batching)
        data = dataset[0]

        return dataset_name, data, in_dim, num_classes

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def build_model(config, in_dim, num_classes):

    hidden = config["hidden_dim"]
    num_layers = config["num_layers"]
    model_name = config["model"].lower()

    if model_name == "gcn":
        model = GCN(in_dim, hidden, num_classes, num_layers)

    elif model_name == "graphsage":
        model = GraphSAGE(in_dim, hidden, num_classes, num_layers)

    elif model_name == "gat":
        model = GAT(in_dim, hidden, num_classes, num_layers)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model


def apply_rewiring(config, data, logger):

    rewiring = config.get("rewiring", "none")

    if rewiring == "none":
        logger.info("No rewiring applied")

    elif rewiring == "virtual_nodes":
        logger.info("Applying virtual node rewiring")
        data = add_virtual_node(data)

    elif rewiring == "curvature":
        logger.info("Applying curvature rewiring")
        data = curvature_rewire(data)

    else:
        raise ValueError(f"Unknown rewiring method: {rewiring}")

    return data


def main(config_path):

    logger = get_logger()

    logger.info(f"Loading experiment config: {config_path}")

    config = load_config(config_path)

    set_seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    dataset_name, data, in_dim, num_classes = load_dataset(config, logger)
    data = apply_rewiring(config, data, logger)
    data = data.to(device)

    model = build_model(config, in_dim, num_classes)
    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"]
    )

    logger.info("Starting training")

    history = train_cora(
        model,
        data,
        optimizer,
        epochs=config["epochs"]
    )

    logger.info("Plotting")

    plot_dir = f"results/plots/{dataset_name}"
    os.makedirs(plot_dir, exist_ok=True)

    plot_path = f"{plot_dir}/{config['experiment_name']}.png"
    plot_training(history, plot_path)

    logger.info("Evaluating model")

    acc = evaluate_cora(model, data)

    results = {
        "experiment": config["experiment_name"],
        "test_accuracy": acc
    }

    table_dir = f"results/tables/{dataset_name}"
    os.makedirs(table_dir, exist_ok=True)

    table_path = f"{table_dir}/{config['experiment_name']}.json"

    with open(table_path, "w") as f:
        json.dump(results, f, indent=4)

    logger.info(f"Test accuracy: {acc:.4f}")


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python main.py <experiment_config.yaml>")
        sys.exit(1)

    config_path = sys.argv[1]

    main(config_path)

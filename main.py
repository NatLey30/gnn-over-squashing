from typing import Dict, Any
import sys
import os
import json
import torch

from src.config import load_config
from src.utils.seed import set_seed
from src.utils.logging import get_logger
from src.utils.plotting import plot_training

from src.experiments import graph_classification, node_classification


def get_layers_dir(config: Dict[str, Any]) -> str:
    """
    Build the directory name corresponding to the number of GNN layers.

    Parameters
    ----------
    config : dict
        Experiment configuration.

    Returns
    -------
    str
        Directory name (e.g. "2layers", "4layers").
    """
    return f"{config['num_layers']}layers"


def save_results_json(
    config: Dict[str, Any],
    dataset_name: str,
    metrics: Dict[str, float]
) -> str:
    """
    Save experiment results and configuration to a JSON file.

    Parameters
    ----------
    config : dict
        Experiment configuration.
    dataset_name : str
        Name of the dataset used.
    metrics : dict
        Evaluation metrics.

    Returns
    -------
    str
        Path to the saved JSON file.
    """

    layers_dir = get_layers_dir(config)

    table_dir = os.path.join("results", "tables", dataset_name, layers_dir)
    os.makedirs(table_dir, exist_ok=True)

    table_path = os.path.join(table_dir, f"{config['experiment_name']}.json")

    results = {
        "experiment": config["experiment_name"],
        "dataset": dataset_name,
        "model": config["model"],
        "rewiring": config.get("rewiring", "none"),
        "num_layers": config["num_layers"],
        "hidden_dim": config["hidden_dim"],
        "lr": config["lr"],
        "weight_decay": config["weight_decay"],
        "epochs": config["epochs"],
        "seed": config["seed"],
        "batch_size": config.get("batch_size", 32),
        **metrics,
    }

    with open(table_path, "w") as f:
        json.dump(results, f, indent=4)

    return table_path


def save_plot(
    config: Dict[str, Any],
    dataset_name: str,
    history: Dict[str, list]
) -> str:
    """
    Save training curves plot.

    Parameters
    ----------
    config : dict
        Experiment configuration.
    dataset_name : str
        Dataset name.
    history : dict
        Training history containing metrics such as loss/accuracy.

    Returns
    -------
    str
        Path to the saved plot.
    """

    layers_dir = get_layers_dir(config)

    plot_dir = os.path.join("results", "plots", dataset_name, layers_dir)
    os.makedirs(plot_dir, exist_ok=True)

    plot_path = os.path.join(plot_dir, f"{config['experiment_name']}.png")

    plot_training(history, plot_path)

    return plot_path


def run_dataset_experiment(
    dataset_name: str,
    config: Dict[str, Any],
    logger,
    device: torch.device
):
    """
    Dispatch the experiment to the correct dataset pipeline.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset.
    config : dict
        Experiment configuration.
    logger : Logger
        Logger instance.
    device : torch.device
        Device used for training.

    Returns
    -------
    Tuple[dict, dict]
        Training history and evaluation metrics.
    """

    if dataset_name in ["cora"]:
        return node_classification.run_experiment(config, logger, device, dataset_name)

    if dataset_name.lower() in ["collab", "dd", "enzymes", "mutag", "proteins"]:
        return graph_classification.run_experiment(config, logger, device, dataset_name)

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def main(config_path: str) -> None:
    """
    Main execution function.

    Parameters
    ----------
    config_path : str
        Path to the experiment configuration YAML file.
    """

    logger = get_logger()
    logger.info(f"Loading experiment config: {config_path}")

    config: Dict[str, Any] = load_config(config_path)

    set_seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    dataset_name = config.get("dataset", "cora").lower()

    history, metrics = run_dataset_experiment(
        dataset_name,
        config,
        logger,
        device
    )

    logger.info("Saving training plots")

    plot_path = save_plot(config, dataset_name, history)

    logger.info("Saving experiment results")

    table_path = save_results_json(config, dataset_name, metrics)

    logger.info(f"Plot saved to: {plot_path}")
    logger.info(f"Results saved to: {table_path}")

    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, float):
            logger.info(f"{metric_name}: {metric_value:.4f}")
        else:
            logger.info(f"{metric_name}: {metric_value}")


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python main.py <experiment_config.yaml>")
        sys.exit(1)

    main(sys.argv[1])

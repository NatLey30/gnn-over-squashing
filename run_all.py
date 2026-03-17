"""
Run a full set of baseline experiments for GNN models.

This script sequentially launches experiments for:
    - GCN
    - GraphSAGE
    - GAT

For each model it runs:
    - Cora node classification
    - ENZYMES graph classification

Each experiment is repeated with different numbers of layers.

Experiments are executed by dynamically generating configuration
files and calling `main.py`.
"""

import subprocess
import yaml
import os


MODELS = ["gcn", "graphsage", "gat"]
DATASETS = ["cora", "enzymes"]
LAYERS = [2, 4, 6, 8]

BASE_CONFIG = {
    "hidden_dim": 64,
    "lr": 0.01,
    "weight_decay": 0.0005,
    "batch_size": 32,
    "rewiring": "none",
    "seed": 42,
}


def build_config(model: str, dataset: str, layers: int) -> dict:
    """
    Build a configuration dictionary for an experiment.
    """

    config = BASE_CONFIG.copy()

    config["experiment_name"] = f"baseline_{model}"
    config["model"] = model
    config["dataset"] = dataset
    config["num_layers"] = layers

    if dataset == "cora":
        config["epochs"] = 200
    else:
        config["epochs"] = 400

    return config


def run_experiment(config: dict) -> None:
    """
    Write a temporary YAML config and launch the experiment.
    """

    os.makedirs("temp_configs", exist_ok=True)

    config_path = f"temp_configs/{config['experiment_name']}_{config['dataset']}.yaml"

    with open(config_path, "w") as f:
        yaml.dump(config, f)

    print("\n=======================================")
    print(f"Running {config['experiment_name']} with {config['num_layers']} layers on {config['dataset']}")
    print("=======================================\n")

    subprocess.run(["python", "main.py", config_path], check=True)


def main() -> None:
    """
    Iterate over all model/dataset/layer combinations.
    """

    for model in MODELS:

        for dataset in DATASETS:

            for layers in LAYERS:

                config = build_config(model, dataset, layers)

                run_experiment(config)


if __name__ == "__main__":
    main()

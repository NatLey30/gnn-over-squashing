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
import numpy as np
import json


MODELS = ["gat"]
# MODELS = ["gcn", "graphsage", "gat"]
DATASETS = [#"cora", 
            # "pubmed"]
            # "enzymes",
            # "mutag",
            "dd"]
            # "proteins"]
            # "zinc"]
            # "qm9"]
LAYERS = [2, 4, 6, 8]
SEEDS = [0, 7, 37, 42]

BASE_CONFIG = {
    "hidden_dim": 64,
    "lr": 0.01,
    "weight_decay": 0.0005,
    "rewiring": "curvature",
    "save_model": False,
}


def build_config(model: str, dataset: str, layers: int) -> dict:
    """
    Build a configuration dictionary for an experiment.
    """

    config = BASE_CONFIG.copy()

    config["experiment_name"] = f"ricci_curvature_{model}"
    config["model"] = model
    config["dataset"] = dataset
    config["num_layers"] = layers

    if dataset == "cora" or dataset == "pubmed":
        config["lr"] = 0.001
        config["epochs"] = 200
        config["dropout"] = 0.5
    elif dataset == "qm9":
        config["lr"] = 0.01
        config["batch_size"] = 512
        config["epochs"] = 30
        config["dropout"] = 0.2
    elif dataset == "zinc":
        config["lr"] = 0.01
        config["batch_size"] = 128
        config["epochs"] = 30
        config["dropout"] = 0.2
    else:
        config["lr"] = 0.001
        config["batch_size"] = 128
        config["epochs"] = 100
        config["dropout"] = 0.5

    # config["epochs"] = 2

    return config


# def run_experiment(config: dict) -> None:
#     """
#     Write a temporary YAML config and launch the experiment.
#     """

#     os.makedirs("configs", exist_ok=True)

#     config_path = f"configs/{config['experiment_name']}_{config['dataset']}.yaml"

#     with open(config_path, "w") as f:
#         yaml.dump(config, f)

#     print("\n=======================================")
#     print(f"Running {config['experiment_name']} with {config['num_layers']} layers on {config['dataset']}")
#     print("=======================================\n")

#     subprocess.run(["python", "main.py", config_path], check=True)

def run_single_seed(config: dict, seed: int) -> dict:
    config = config.copy()
    config["seed"] = seed

    os.makedirs("configs", exist_ok=True)

    config_path = f"configs/{config['experiment_name']}_{config['dataset']}_layer{config['num_layers']}_seed{seed}.yaml"

    with open(config_path, "w") as f:
        yaml.dump(config, f)

    # print("\n=======================================")
    # print(f"Running {config['experiment_name']} with {config['num_layers']} layers on {config['dataset']} with seed {config['seed']}")
    # print("=======================================\n")

    subprocess.run(["python", "main.py", config_path], check=True)

    # cargar resultados
    layers_dir = f"{config['num_layers']}layers"
    result_path = os.path.join(
        "results", "tables",
        config["dataset"],
        layers_dir,
        f"{config['experiment_name']}_seed{seed}.json"
    )

    with open(result_path, "r") as f:
        results = json.load(f)

    return results

def save_aggregated_results(config, dataset, aggregated):

    layers_dir = f"{config['num_layers']}layers"

    out_dir = os.path.join("results", "aggregated", dataset, layers_dir)
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, f"{config['experiment_name']}.json")

    with open(out_path, "w") as f:
        json.dump(aggregated, f, indent=4)


def main() -> None:
    """
    Iterate over all model/dataset/layer combinations.
    """

    # for model in MODELS:

    #     for dataset in DATASETS:

    #         for layers in LAYERS:

    #             config = build_config(model, dataset, layers)

    #             run_experiment(config)

    for dataset in DATASETS:
        print("\n=======================================")
        print(f"Running {dataset}")
        print("=======================================\n")
        for model in MODELS:
            for layers in LAYERS:
                print(f"--- Model {model} with {layers} layers")

                config = build_config(model, dataset, layers)

                seed_results = []

                for seed in SEEDS:
                    res = run_single_seed(config, seed)
                    seed_results.append(res)

                # --- calcular mean/std ---
                metrics_keys = [
                    k for k in seed_results[0].keys()
                    if isinstance(seed_results[0][k], float)
                ]

                aggregated = {}

                for k in metrics_keys:
                    values = [r[k] for r in seed_results]
                    aggregated[f"{k}_mean"] = float(np.mean(values))
                    aggregated[f"{k}_std"] = float(np.std(values))

                # guardar agregados
                save_aggregated_results(config, dataset, aggregated)


if __name__ == "__main__":
    main()

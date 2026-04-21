import os
import json
import csv
from typing import Dict, Any, Tuple, List, DefaultDict
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


BASE_DIR: str = "results/aggregated"
SAVE_BASE: str = "results/plots"

METHOD_COLORS = {
    "baseline": "#4D4D4D",   # dark gray
    "virtual": "#4C72B0",    # muted blue
    "ricci": "#C44E52",      # muted red
    "other": "#999999"
}

plt.style.use("seaborn-v0_8-paper")

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9
})

# --------------------------------------------------
# UTILITIES
# --------------------------------------------------

def ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def safe_load_json(path: str) -> Dict[str, Any] | None:
    """
    Safely load a JSON file.

    Returns None if the file is corrupted or unreadable.
    """
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def get_metric(metrics: Dict[str, Any]) -> Tuple[str, float] | None:
    """
    Extract metric name and value from a metrics dict.

    Returns:
        (metric_name, value) or None if not found.
    """
    if "test_accuracy_mean" in metrics:
        return "accuracy", metrics["test_accuracy_mean"]
    elif "test_mae_mean" in metrics:
        return "mae", metrics["test_mae_mean"]
    return None


def split_name(name: str) -> Tuple[str, str]:
    """
    Split experiment name into (method, model).

    Examples:
        baseline_gcn -> (baseline, gcn)
        ricci_curvature_gat -> (ricci, gat)
        virtual_nodes_graphsage -> (virtual, graphsage)
    """
    if name.startswith("baseline_"):
        return "baseline", name.replace("baseline_", "")
    elif name.startswith("ricci_curvature_"):
        return "ricci", name.replace("ricci_curvature_", "")
    elif name.startswith("virtual_nodes_"):
        return "virtual", name.replace("virtual_nodes_", "")
    return "other", name


# --------------------------------------------------
# DATA LOADING
# --------------------------------------------------

def load_dataset_results(dataset_path: str) -> Dict[int, Dict[str, Dict[str, Any]]]:
    """
    Load results for a dataset.

    Structure:
        dict[layer][experiment_name] = metrics

    Missing or corrupted files are skipped.
    """
    data: DefaultDict[int, Dict[str, Dict[str, Any]]] = defaultdict(dict)

    for layer_dir in os.listdir(dataset_path):
        if not layer_dir.endswith("layers"):
            continue

        try:
            num_layers: int = int(layer_dir.replace("layers", ""))
        except ValueError:
            continue

        layer_path = os.path.join(dataset_path, layer_dir)

        for file in os.listdir(layer_path):
            if not file.endswith(".json"):
                continue

            path = os.path.join(layer_path, file)
            metrics = safe_load_json(path)

            if metrics is None:
                continue

            name: str = file.replace(".json", "")
            data[num_layers][name] = metrics

    return data


# --------------------------------------------------
# PLOTS
# --------------------------------------------------

def plot_per_model(
    data: Dict[int, Dict[str, Dict[str, Any]]],
    dataset: str,
    save_dir: str
) -> None:
    """
    Plot performance vs layers for each model and method.

    Skips missing values automatically.
    """
    model_data: DefaultDict[str, DefaultDict[str, List[Tuple[int, float]]]] = defaultdict(lambda: defaultdict(list))

    metric_name: str | None = None

    for layers, configs in data.items():
        for name, metrics in configs.items():
            metric = get_metric(metrics)
            if metric is None:
                continue

            metric_name, value = metric
            method, model = split_name(name)

            model_data[model][method].append((layers, value))

    for model, methods in model_data.items():
        plt.figure()

        for method, values in methods.items():
            if len(values) == 0:
                continue

            values = sorted(values)
            x = [v[0] for v in values]
            y = [v[1] for v in values]

            color = METHOD_COLORS.get(method, "#999999")
            plt.plot(x, y, marker="o", label=method, color=color)

        if metric_name is None:
            continue

        plt.xlabel("Layers")
        plt.ylabel(metric_name.upper())
        plt.title(f"{dataset} - {model}")
        plt.legend()
        plt.grid()

        plt.savefig(os.path.join(save_dir, f"{model}_vs_layers.png"))
        plt.close()


def plot_methods_comparison(
    data: Dict[int, Dict[str, Dict[str, Any]]],
    dataset: str,
    save_dir: str
) -> None:
    """
    Plot average performance per method across models.
    """
    method_data: DefaultDict[str, DefaultDict[int, List[float]]] = defaultdict(lambda: defaultdict(list))

    metric_name: str | None = None

    for layers, configs in data.items():
        for name, metrics in configs.items():
            metric = get_metric(metrics)
            if metric is None:
                continue

            metric_name, value = metric
            method, _ = split_name(name)

            method_data[method][layers].append(value)

    plt.figure()

    for method, layer_dict in method_data.items():
        xs: List[int] = []
        ys: List[float] = []

        for layer, values in sorted(layer_dict.items()):
            if len(values) == 0:
                continue

            xs.append(layer)
            ys.append(float(np.mean(values)))

        if len(xs) == 0:
            continue

        color = METHOD_COLORS.get(method, "#999999")
        plt.plot(xs, ys, marker="o", label=method, color=color)

    if metric_name is None:
        return

    plt.xlabel("Layers")
    plt.ylabel(metric_name.upper())
    plt.title(f"{dataset} - Method comparison")
    plt.legend()
    plt.grid()

    plt.savefig(os.path.join(save_dir, "methods_comparison.png"))
    plt.close()


def plot_ranking(
    data: Dict[int, Dict[str, Dict[str, Any]]],
    dataset: str,
    save_dir: str
) -> None:
    """
    Plot ranking of all configurations.
    """
    results: List[Tuple[str, int, float]] = []
    metric_name: str | None = None

    for layers, configs in data.items():
        for name, metrics in configs.items():
            metric = get_metric(metrics)
            if metric is None:
                continue

            metric_name, value = metric
            results.append((name, layers, value))

    if len(results) == 0 or metric_name is None:
        return

    results = sorted(results, key=lambda x: x[2], reverse=True)

    names = [f"{n} ({l})" for n, l, _ in results]
    values = [v for _, _, v in results]

    plt.figure(figsize=(10, 6))
    colors = []
    for name, _, _ in results:
        method, _ = split_name(name)
        colors.append(METHOD_COLORS.get(method, "#999999"))

    plt.barh(names, values, color=colors)
    plt.xlabel(metric_name.upper())
    plt.title(f"{dataset} - Ranking")
    plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "ranking.png"))
    plt.close()


def plot_heatmap(
    data: Dict[int, Dict[str, Dict[str, Any]]],
    dataset: str,
    save_dir: str
) -> None:
    """
    Plot heatmap of configurations vs layers.

    Missing values are filled with NaN.
    """
    matrix: DefaultDict[str, Dict[int, float]] = defaultdict(dict)

    metric_name: str | None = None

    for layers, configs in data.items():
        for name, metrics in configs.items():
            metric = get_metric(metrics)
            if metric is None:
                continue

            metric_name, value = metric
            method, model = split_name(name)

            key = f"{method}_{model}"
            matrix[key][layers] = value

    if len(matrix) == 0 or metric_name is None:
        return

    keys = sorted(matrix.keys())
    layers = sorted(data.keys())

    heat = np.full((len(keys), len(layers)), np.nan)

    for i, k in enumerate(keys):
        for j, l in enumerate(layers):
            if l in matrix[k]:
                heat[i, j] = matrix[k][l]

    plt.figure(figsize=(8, 6))
    sns.heatmap(heat, annot=True, xticklabels=layers, yticklabels=keys)

    plt.xlabel("Layers")
    plt.ylabel("Config")
    plt.title(f"{dataset} - Heatmap")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "heatmap.png"))
    plt.close()


def plot_delta_vs_baseline(
    data: Dict[int, Dict[str, Dict[str, Any]]],
    dataset: str,
    save_dir: str
) -> None:
    """
    Plot improvement over baseline (delta) vs layers.
    """

    delta_data: DefaultDict[str, DefaultDict[str, List[Tuple[int, float]]]] = defaultdict(lambda: defaultdict(list))

    for layers, configs in data.items():
        for name, metrics in configs.items():
            metric = get_metric(metrics)
            if metric is None:
                continue

            _, value = metric
            method, model = split_name(name)

            if method == "baseline":
                continue

            baseline_key = f"baseline_{model}"

            if baseline_key not in configs:
                continue

            base_metric = get_metric(configs[baseline_key])
            if base_metric is None:
                continue

            _, base_value = base_metric

            delta = value - base_value
            delta_data[model][method].append((layers, delta))

    for model, methods in delta_data.items():
        plt.figure()

        for method, values in methods.items():
            if len(values) == 0:
                continue

            values = sorted(values)
            x = [v[0] for v in values]
            y = [v[1] for v in values]

            color = METHOD_COLORS.get(method, "#999999")
            plt.plot(x, y, marker="o", label=method, color=color)

        plt.axhline(0, linestyle="--", color="#7F7F7F", linewidth=1)
        plt.xlabel("Layers")
        plt.ylabel("Delta vs baseline")
        plt.title(f"{dataset} - Delta ({model})")
        plt.legend()
        plt.grid()

        plt.savefig(os.path.join(save_dir, f"{model}_delta.png"))
        plt.close()


def plot_performance_drop(
    data: Dict[int, Dict[str, Dict[str, Any]]],
    dataset: str,
    save_dir: str
) -> None:
    """
    Plot performance drop relative to shallow model (min layers).
    """

    model_data: DefaultDict[str, DefaultDict[str, List[Tuple[int, float]]]] = defaultdict(lambda: defaultdict(list))

    for layers, configs in data.items():
        for name, metrics in configs.items():
            metric = get_metric(metrics)
            if metric is None:
                continue

            _, value = metric
            method, model = split_name(name)

            model_data[f"{method}_{model}"]["values"].append((layers, value))

    for config, d in model_data.items():
        values = sorted(d["values"])
        if len(values) == 0:
            continue

        base_layer, base_value = values[0]

        x = []
        y = []

        for layer, value in values:
            x.append(layer)
            y.append(value - base_value)

        plt.figure()
        color = METHOD_COLORS.get(method, "#999999")
        plt.plot(x, y, marker="o", label=method, color=color)
        plt.axhline(0, linestyle="--", color="#7F7F7F", linewidth=1)

        plt.xlabel("Layers")
        plt.ylabel("Performance drop")
        plt.title(f"{dataset} - {config}")

        plt.grid()
        plt.savefig(os.path.join(save_dir, f"{config}_drop.png"))
        plt.close()


def build_summary_csv(
    all_data: Dict[str, Dict[int, Dict[str, Dict[str, Any]]]],
    save_path: str = "results/summary.csv"
) -> None:
    """
    Build a summary CSV with format:
        rows = configurations
        cols = datasets
        values = mean ± std
    """

    table: DefaultDict[str, Dict[str, str]] = defaultdict(dict)

    for dataset, data in all_data.items():
        for layers, configs in data.items():
            for name, metrics in configs.items():

                metric_name, mean = get_metric(metrics) or (None, None)
                if metric_name is None:
                    continue

                std_key = f"test_{metric_name}_std"
                std = metrics.get(std_key, 0.0)

                value_str = f"{mean:.4f} ± {std:.4f}"

                key = f"{name}_{layers}L"
                table[key][dataset] = value_str

    datasets = sorted(all_data.keys())
    rows = sorted(table.keys())

    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(["Config"] + datasets)

        for row in rows:
            row_values = [table[row].get(ds, "") for ds in datasets]
            writer.writerow([row] + row_values)

    print(f"[INFO] Saved summary CSV to {save_path}")


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main() -> None:
    all_data = {}

    for dataset in os.listdir(BASE_DIR):
        dataset_path = os.path.join(BASE_DIR, dataset)

        if not os.path.isdir(dataset_path):
            continue

        print(f"[INFO] Processing {dataset}")

        data = load_dataset_results(dataset_path)

        if len(data) == 0:
            continue

        all_data[dataset] = data

        save_dir = os.path.join(SAVE_BASE, dataset, "analysis")
        ensure_dir(save_dir)

        plot_per_model(data, dataset, save_dir)
        plot_methods_comparison(data, dataset, save_dir)
        plot_ranking(data, dataset, save_dir)
        plot_heatmap(data, dataset, save_dir)
        plot_delta_vs_baseline(data, dataset, save_dir)
        plot_performance_drop(data, dataset, save_dir)

    build_summary_csv(all_data)


if __name__ == "__main__":
    main()

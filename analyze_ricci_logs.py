import os
import re
import time
from typing import Dict, List, Optional
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# REGEX
# -----------------------------
LOG_PATTERN = re.compile(
    r"nodes=(?P<nodes>\d+)\s*\|\s*edges\s*(?P<edges_before>\d+)->(?P<edges_after>\d+)\s*\(\+(?P<edges_added>\d+)\)\s*\|\s*density\s*(?P<density_before>[0-9.]+)->(?P<density_after>[0-9.]+)\s*\|\s*avg_degree\s*(?P<deg_before>[0-9.]+)->(?P<deg_after>[0-9.]+)"
)


# -----------------------------
# PARSING
# -----------------------------
def parse_line(line: str) -> Optional[Dict[str, float]]:
    """Parse a single rewiring log line including timestamp."""

    # Split timestamp
    try:
        timestamp_str, rest = line.split(" | INFO | ", 1)
        timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S,%f")
    except ValueError:
        return None

    match = LOG_PATTERN.search(rest)
    if match is None:
        return None

    data = match.groupdict()

    return {
        "timestamp": timestamp,
        "nodes": int(data["nodes"]),
        "edges_before": int(data["edges_before"]),
        "edges_after": int(data["edges_after"]),
        "edges_added": int(data["edges_added"]),
        "density_before": float(data["density_before"]),
        "density_after": float(data["density_after"]),
        "degree_before": float(data["deg_before"]),
        "degree_after": float(data["deg_after"]),
    }


def parse_log_file(path: str) -> pd.DataFrame:
    """Parse all valid lines from a dataset log file."""
    rows: List[Dict[str, float]] = []

    with open(path, "r") as f:
        for line in f:
            parsed = parse_line(line)
            if parsed is not None:
                rows.append(parsed)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Sort by time (important)
    df = df.sort_values("timestamp")

    # Compute time differences
    df["delta_time_sec"] = df["timestamp"].diff().dt.total_seconds()

    # Derived metrics
    df["edges_ratio"] = df["edges_after"] / df["edges_before"]
    df["density_delta"] = df["density_after"] - df["density_before"]
    df["degree_delta"] = df["degree_after"] - df["degree_before"]
    return df


# -----------------------------
# SUMMARY
# -----------------------------
def compute_summary(df: pd.DataFrame, dataset: str) -> Dict:
    """Compute aggregated stats for a dataset."""
    if df.empty:
        return {}

    total_time = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).total_seconds()

    return {
        "dataset": dataset,
        "num_graphs": len(df),

        # Structural changes
        "edges_added_mean": df["edges_added"].mean(),
        "edges_added_std": df["edges_added"].std(),
        "edges_ratio_mean": df["edges_ratio"].mean(),
        "density_delta_mean": df["density_delta"].mean(),
        "degree_delta_mean": df["degree_delta"].mean(),

        # Distribution info
        "edges_added_p95": df["edges_added"].quantile(0.95),
        "edges_added_max": df["edges_added"].max(),

        # Correlation
        "corr_nodes_edges_added": df["nodes"].corr(df["edges_added"]),

        # REAL timing
        "time_total_sec": total_time,
        "time_per_graph_ms": (total_time / len(df)),
    }

# -----------------------------
# PLOTTING
# -----------------------------
COLORS = {
    "edges": "#4C72B0",
    "density": "#55A868",
    "degree": "#C44E52",
}


def plot_distributions(df: pd.DataFrame, dataset: str, out_dir: str) -> None:
    """Plot histogram of edges added."""
    plt.figure()
    plt.hist(df["edges_added"], bins=30)
    plt.xlabel("Edges added")
    plt.ylabel("Frequency")
    plt.title(f"{dataset} - Edges added distribution")
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f"{dataset}_edges_added_hist.png"))
    plt.close()


def plot_scatter(df: pd.DataFrame, dataset: str, out_dir: str) -> None:
    """Plot nodes vs edges added (bottleneck intuition)."""
    plt.figure()
    plt.scatter(df["nodes"], df["edges_added"], alpha=0.6)
    plt.xlabel("Number of nodes")
    plt.ylabel("Edges added")
    plt.title(f"{dataset} - Rewiring vs graph size")
    plt.tight_layout()

    plt.savefig(os.path.join(out_dir, f"{dataset}_nodes_vs_edges.png"))
    plt.close()


# -----------------------------
# LATEX EXPORT
# -----------------------------
def save_latex_table(df: pd.DataFrame, path: str) -> None:
    """Save summary table in LaTeX format."""
    latex = df.to_latex(
        index=False,
        float_format="%.4f",
        caption="Structural effect of Ricci rewiring",
        label="tab:rewiring_summary",
    )

    with open(path, "w") as f:
        f.write(latex)


# -----------------------------
# MAIN ANALYSIS
# -----------------------------
def analyze_folder(folder: str) -> pd.DataFrame:
    """Full pipeline: parse, analyze, plot."""
    summaries: List[Dict] = []

    for file_name in os.listdir(folder):
        if not file_name.endswith(".txt"):
            continue

        dataset = file_name.replace(".txt", "")
        path = os.path.join(folder, file_name)

        print(f"\nProcessing {dataset}...")

        start = time.time()
        df = parse_log_file(path)
        elapsed = time.time() - start

        if df.empty:
            print(f"[WARNING] Skipping {dataset} (no valid data)")
            continue

        # Save per-graph raw data
        raw_out = f"results/rewiring_raw/{dataset}.csv"
        os.makedirs(os.path.dirname(raw_out), exist_ok=True)
        df.to_csv(raw_out, index=False)

        # Plots
        plot_dir = "results/plots/rewiring"
        plot_distributions(df, dataset, plot_dir)
        plot_scatter(df, dataset, plot_dir)

        # Summary
        summary = compute_summary(df, dataset)
        summaries.append(summary)

        print(f"  graphs: {len(df)}")
        print(f"  time: {elapsed:.2f}s ({summary['time_per_graph_ms']:.2f} ms/graph)")

    return pd.DataFrame(summaries)


def main():
    total_start = time.time()

    folder = "results/logging"
    df = analyze_folder(folder)

    if df.empty:
        print("No results found.")
        return

    df = df.sort_values("dataset")

    # Save CSV
    csv_path = "results/rewiring_summary.csv"
    df.to_csv(csv_path, index=False)

    # Save LaTeX
    latex_path = "results/rewiring_summary.tex"
    save_latex_table(df, latex_path)

    total_time = time.time() - total_start

    print("\n=== FINAL SUMMARY ===")
    print(df)
    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved LaTeX: {latex_path}")
    print(f"Total execution time: {total_time:.2f}s")


if __name__ == "__main__":
    main()

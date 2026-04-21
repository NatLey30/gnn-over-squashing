# GNN Over-Squashing and Graph Rewiring

## Overview

In this repo we study **over-squashing in Graph Neural Networks (GNNs)**.

Basically, when graphs get complex, information from distant nodes gets compressed too much as it flows through the network. This can hurt performance, especially when we increase depth.

The main idea of the project is:

> How does graph structure affect over-squashing, and can we mitigate it using rewiring techniques like Ricci curvature?

So instead of just training models, we try to **understand what’s going on structurally** and whether modifying the graph helps.

---

## What we are trying to see

We focus on three main things:

- How performance changes as we increase the number of layers
- Whether structural properties (like bottlenecks or edge betweenness) explain performance drops
- Whether **rewiring (Ricci curvature)** improves message passing

We are not trying to beat SOTA. The goal is to **analyze behavior and extract insights**.

---

## Project Structure

```text
.
├── configs/               # Experiment configs (yaml)
├── data/                  # Raw and processed datasets
│   └── rewired/           # Rewired versions of datasets aplicado
├── results/
│   ├── aggregated/        # Aggregated results by seeds (json)
│   ├── histories/         # Train histories (json)
│   ├── plots/             # Generated plots
│   ├── tables/            # Final metrics (json)
├── src/
│   ├── models/            # GNN models (GCN, GAT, GraphSAGE)
│   ├── training/          # Training loops
│   ├── rewiring/          # Ricci curvature + graph rewiring
│   ├── utils/             # Helpers (logging, saving, etc.)
│   └── datasets/          # Dataset loading
├── analysis.py
├── analyze_ricci_logs.py
├── main.py                # Entry point for experiments
├── requirements.txt
├── README.md
└── run_all.py
```

---

## Installation

Clone the repo and install dependencies:

```bash
git clone <repo_url>
cd <repo_name>

python -m venv venv
source venv/bin/activate  # or conda activate

pip install -r requirements.txt
```
If you are using GPU, make sure PyTorch is installed with CUDA.

---
## Datasets

We use different datasets depending on the task.

### Node Classification

- **Cora**
- **PubMed**

These are citation networks where nodes are papers and edges are citations.

---

### Graph Classification

- **MUTAG**
- **PROTEINS**
- **DD**
- **ENZYMES**

Each graph represents a molecule or structure, and we classify the whole graph.

---

### Graph Regression

- **ZINC**
- **QM9**

Here the task is to predict continuous molecular properties.

---

## Experiments

### Models

We evaluate standard message-passing GNNs:

- GCN  
- GraphSAGE  
- GAT  

These models allow us to study how depth affects performance under different architectures.

---

## Experiments

### Models

We evaluate standard message-passing GNNs:

- GCN  
- GraphSAGE  
- GAT  
These models allow us to study how depth affects performance under different architectures.

---

### Experimental Setup

Typical configuration:

- Number of layers: `[2, 4, 6, 8]`
- Seeds: `[0, 7, 37, 42]`
- Hidden dimension: Depending on the dataset
- Optimizer: Adam
- Learning rate: 0.01 (classification) / 0.001 (regression)
- Weight decay: 5e-4
- Dropout: 0.2–0.5 depending on the model

Metrics:

- **Node / Graph Classification** → Accuracy  
- **Graph Regression** → MAE (and sometimes MSE)

We report **mean ± std across seeds**.

---

### What we measure

#### 1. Performance vs Depth

We analyze how performance evolves as we increase the number of layers.

- Does performance degrade with depth?
- Is there an optimal number of layers?
- Do different models behave differently?

This is the main signal of over-squashing.

---

#### 2. Structural Analysis

We compute graph-level metrics to understand potential bottlenecks:

- **Edge betweenness**
  - Measures how much shortest-path traffic goes through each edge
  - High values → possible bottlenecks

This helps us relate **graph structure → model performance**.

---

#### 3. Rewiring Experiments (Ricci Curvature)

We apply **Ricci curvature-based rewiring** to modify graph topology.

Main idea:

- Edges with negative curvature → bottlenecks
- Rewire graph to improve information flow

We then compare:

- Original graph vs rewired graph  
- Performance vs depth in both cases  

---

#### 4. Virtual Node Experiments

We also evaluate the use of **virtual nodes** as an alternative way to mitigate over-squashing.

Idea:

- Add a global node connected to all nodes in the graph
- This creates a shortcut for long-range communication
- Helps reduce the need for deep message passing

We compare:

- Standard GNNs vs GNNs with virtual nodes  
- Behavior across different depths  
- Interaction with graph structure  

This provides a complementary approach to rewiring, since instead of modifying edges locally, it introduces a global communication mechanism.

---

#### 3. Rewiring Experiments

We apply **Ricci curvature-based rewiring** to modify graph topology.

Main idea:

- Edges with negative curvature → bottlenecks
- Rewire graph to improve information flow

We then compare:

- Original graph vs rewired graph  
- Performance vs depth in both cases  

---

## Running Experiments

Basic usage:

```bash
python main.py <config_path>
```
---

## Running full sweeps
Experiments are typically run over:

* Multiple models
* Multiple layer values
* Multiple random seeds

This is handled via config files and loops inside the training pipeline.

---

## Rewiring (Ricci Curvature)
We implement Ricci curvature-based graph rewiring.

**Steps:**
1. Compute curvature for each edge
2. Identify edges acting as bottlenecks
3. Rewire graph (add/remove edges)
4. Train models on modified graph

Rewired datasets are stored in: `data/rewired/`

---

## Results
All outputs are automatically saved.

### Tables (final metrics)
`results/tables/<dataset>/<layers_dir>/`

Each file contains:
```json
{
    "experiment": "...",
    "dataset": "...",
    "model": "...",
    "rewiring": "...",
    "num_layers": ...,
    "hidden_dim": ...,
    "lr": ...,
    "weight_decay": ...,
    "epochs": ...,
    "seed": ...,
    "batch_size": ...,
    "test_accuracy": ...
}
```
### Histories (training curves)
`results/histories/<dataset>/<experiment_name>/`

Includes:
* Train loss
* Validation loss
* Accuracy / MAE

### Plots
`results/plots/`

Includes:
* Performance vs layers
* Structural metrics
* Comparison between original and rewired graphs

### Reproducibility
We ensure reproducibility by:
* Fixing random seeds
* Logging configurations
* Saving all results and histories

## Future Work
Possible extensions:
* Spectral rewiring methods
* Deeper curvature analysis
* More expressive GNN architectures
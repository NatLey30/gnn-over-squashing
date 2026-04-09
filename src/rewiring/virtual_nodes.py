"""
virtual_nodes.py
----------------
Graph rewiring strategy: Virtual Node Injection.

A virtual node is a synthetic node connected bidirectionally to every real node
in the graph. This creates O(1)-length paths between any pair of nodes (all
nodes are at most 2 hops apart via the virtual node), which directly alleviates
over-squashing by shortening long-range information pathways.

Supports three task types:
    - "node"  : node classification (y per node, train/val/test masks present)
    - "graph" : graph classification (y is a single integer label per graph)
    - "regression" : graph regression (y is a continuous value per graph)
"""

import torch
from torch_geometric.data import Data


def _detect_task(data: Data) -> str:
    """Heuristically detect the task type from the Data object.

    Returns one of: "node", "graph", "regression".
    Detection logic:
      - If train_mask / val_mask / test_mask are present  → "node"
      - If y is integer dtype                             → "graph"
      - Otherwise                                         → "regression"
    """
    has_masks = all(
        hasattr(data, attr) and getattr(data, attr) is not None
        for attr in ("train_mask", "val_mask", "test_mask")
    )
    if has_masks:
        return "node"
    if data.y is not None and not data.y.is_floating_point():
        return "graph"
    return "regression"


def add_virtual_node(data: Data, task: str = "auto") -> Data:
    """Add a single virtual node connected to all real nodes in the graph.

    The virtual node is appended as the last node in the graph. It is connected
    bidirectionally to every existing node, effectively acting as a global
    aggregator that reduces the effective diameter of the graph to 2.

    Behaviour differs by task type:

    Node classification (``task="node"``)
        The virtual node is excluded from every loss/metric computation.
        ``data.y`` (shape ``[N]``) gets a dummy label 0 appended.
        All three split masks get a ``False`` appended.

    Graph classification (``task="graph"``)
        ``data.y`` holds a single label for the whole graph and must NOT be
        modified. Split masks are not expected and are not created.

    Graph regression (``task="regression"``)
        Same as graph classification: ``data.y`` is a graph-level target value
        and is left completely untouched.

    Args:
        data: A PyTorch Geometric ``Data`` object representing the input graph.
            Required attributes for all tasks:
            - ``x``          – node feature matrix, shape ``[N, F]``
            - ``edge_index`` – COO edge index, shape ``[2, E]``
            - ``y``          – labels / targets

            Additional attributes required for node classification:
            - ``train_mask``, ``val_mask``, ``test_mask`` – boolean masks ``[N]``

        task: One of ``"node"``, ``"graph"``, ``"regression"``, or ``"auto"``
            (default).  When ``"auto"``, the task is inferred from the Data
            object using :func:`_detect_task`.

    Returns:
        A new ``Data`` object with ``N+1`` nodes.  All original edges are
        preserved; ``2*N`` new directed edges (N from virtual → real, N from
        real → virtual) are appended to ``edge_index``.

    Raises:
        ValueError: If ``task`` is not one of the accepted values.

    Example (node classification):
        >>> from torch_geometric.datasets import Planetoid
        >>> dataset = Planetoid(root='/tmp/Cora', name='Cora')
        >>> data = add_virtual_node(dataset[0], task="node")
        >>> data.num_nodes == dataset[0].num_nodes + 1
        True

    Example (graph classification / regression):
        >>> from torch_geometric.datasets import TUDataset
        >>> dataset = TUDataset(root='/tmp/MUTAG', name='MUTAG')
        >>> data = add_virtual_node(dataset[0], task="graph")
        >>> data.num_nodes == dataset[0].num_nodes + 1
        True
        >>> data.y.shape == dataset[0].y.shape   # graph-level label untouched
        True
    """
    valid_tasks = {"node", "graph", "regression", "auto"}
    if task not in valid_tasks:
        raise ValueError(
            f"Unknown task '{task}'. Choose one of {valid_tasks}."
        )

    if task == "auto":
        task = _detect_task(data)

    num_nodes: int = data.num_nodes   # N  (original node count)
    virtual_id: int = num_nodes        # index of the new virtual node

    # ------------------------------------------------------------------
    # 1. Extend node features: append a zero row for the virtual node.
    # ------------------------------------------------------------------
    # Shape: [N, F] → [N+1, F]
    virtual_feat = torch.zeros(1, data.x.size(1), dtype=data.x.dtype)
    x_new = torch.cat([data.x, virtual_feat], dim=0)

    # ------------------------------------------------------------------
    # 2. Build new bidirectional edges between the virtual node and every
    #    real node.
    #    • virtual → real : (virtual_id, i)  for i in 0..N-1
    #    • real → virtual : (i, virtual_id)  for i in 0..N-1
    # ------------------------------------------------------------------
    real_node_ids = torch.arange(num_nodes, dtype=torch.long)
    virtual_ids   = torch.full((num_nodes,), virtual_id, dtype=torch.long)

    virtual_to_real = torch.stack([virtual_ids,    real_node_ids], dim=0)  # [2, N]
    real_to_virtual = torch.stack([real_node_ids,  virtual_ids],   dim=0)  # [2, N]

    # Shape: [2, E + 2*N]
    edge_index_new = torch.cat(
        [data.edge_index, virtual_to_real, real_to_virtual], dim=1
    )

    # ------------------------------------------------------------------
    # 3. Handle labels and masks depending on task type.
    # ------------------------------------------------------------------
    if task == "node":
        # y has shape [N] — append a dummy label excluded by masks.
        dummy_label = torch.zeros(1, dtype=data.y.dtype)
        y_new = torch.cat([data.y, dummy_label], dim=0)

        false_flag = torch.zeros(1, dtype=torch.bool)
        train_mask_new = torch.cat([data.train_mask, false_flag], dim=0)
        val_mask_new   = torch.cat([data.val_mask,   false_flag], dim=0)
        test_mask_new  = torch.cat([data.test_mask,  false_flag], dim=0)

        rewired_data = Data(
            x=x_new,
            edge_index=edge_index_new,
            y=y_new,
            train_mask=train_mask_new,
            val_mask=val_mask_new,
            test_mask=test_mask_new,
        )

    else:
        # "graph" or "regression":
        # y is a graph-level label/target — do NOT modify it.
        # No split masks to extend (they live at the dataset/loader level).
        rewired_data = Data(
            x=x_new,
            edge_index=edge_index_new,
            y=data.y,
        )

    # ------------------------------------------------------------------
    # 4. Propagate any extra attributes that are not task-critical
    #    (e.g. edge_attr, pos, batch) to avoid silently dropping them.
    # ------------------------------------------------------------------
    skip_attrs = {"x", "edge_index", "y", "train_mask", "val_mask", "test_mask"}
    for key, value in data:
        if key not in skip_attrs:
            setattr(rewired_data, key, value)

    return rewired_data

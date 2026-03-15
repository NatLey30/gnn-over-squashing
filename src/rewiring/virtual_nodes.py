"""
virtual_nodes.py
----------------
Graph rewiring strategy: Virtual Node Injection.

A virtual node is a synthetic node connected bidirectionally to every real node
in the graph. This creates O(1)-length paths between any pair of nodes (all
nodes are at most 2 hops apart via the virtual node), which directly alleviates
over-squashing by shortening long-range information pathways.
"""

import torch
from torch_geometric.data import Data


def add_virtual_node(data: Data) -> Data:
    """Add a single virtual node connected to all real nodes in the graph.

    The virtual node is appended as the last node in the graph. It is connected
    bidirectionally to every existing node, effectively acting as a global
    aggregator that reduces the effective diameter of the graph to 2.

    The node feature for the virtual node is initialised to a zero vector of
    the same width as the existing node features.  All three split masks
    (train / val / test) are extended with ``False`` so that the virtual node
    is never included in any loss or evaluation computation, and a dummy label
    of 0 is appended to ``data.y`` for the same reason.

    Args:
        data: A PyTorch Geometric ``Data`` object representing the input graph.
            Expected attributes:
            - ``x``          – node feature matrix, shape ``[N, F]``
            - ``edge_index`` – COO edge index, shape ``[2, E]``
            - ``y``          – node labels, shape ``[N]``
            - ``train_mask`` – boolean mask, shape ``[N]``
            - ``val_mask``   – boolean mask, shape ``[N]``
            - ``test_mask``  – boolean mask, shape ``[N]``

    Returns:
        A new ``Data`` object with ``N+1`` nodes.  All original edges are
        preserved; ``2*N`` new directed edges (N from virtual → real, N from
        real → virtual) are appended to ``edge_index``.

    Example:
        >>> from torch_geometric.datasets import Planetoid
        >>> dataset = Planetoid(root='/tmp/Cora', name='Cora')
        >>> data = add_virtual_node(dataset[0])
        >>> data.num_nodes == dataset[0].num_nodes + 1
        True
    """
    num_nodes: int = data.num_nodes  # N  (original node count)
    virtual_id: int = num_nodes       # index of the new virtual node

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
    virtual_ids = torch.full((num_nodes,), virtual_id, dtype=torch.long)

    # Each tensor is shape [2, N]; concatenate along the edge dimension.
    virtual_to_real = torch.stack([virtual_ids, real_node_ids], dim=0)  # [2, N]
    real_to_virtual = torch.stack([real_node_ids, virtual_ids], dim=0)  # [2, N]

    # Combine original edges with the new virtual edges.
    # Shape: [2, E + 2*N]
    edge_index_new = torch.cat(
        [data.edge_index, virtual_to_real, real_to_virtual], dim=1
    )

    # ------------------------------------------------------------------
    # 3. Extend labels: append a dummy label (0) for the virtual node.
    #    The mask extensions below ensure it is never used in training or
    #    evaluation.
    # ------------------------------------------------------------------
    dummy_label = torch.zeros(1, dtype=data.y.dtype)
    y_new = torch.cat([data.y, dummy_label], dim=0)

    # ------------------------------------------------------------------
    # 4. Extend split masks with False so the virtual node is excluded
    #    from every loss / metric computation.
    # ------------------------------------------------------------------
    false_flag = torch.zeros(1, dtype=torch.bool)

    train_mask_new = torch.cat([data.train_mask, false_flag], dim=0)
    val_mask_new = torch.cat([data.val_mask,   false_flag], dim=0)
    test_mask_new = torch.cat([data.test_mask,  false_flag], dim=0)

    # ------------------------------------------------------------------
    # 5. Assemble and return the updated Data object.
    # ------------------------------------------------------------------
    rewired_data = Data(
        x=x_new,
        edge_index=edge_index_new,
        y=y_new,
        train_mask=train_mask_new,
        val_mask=val_mask_new,
        test_mask=test_mask_new,
    )

    return rewired_data

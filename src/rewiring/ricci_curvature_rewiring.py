"""
ricci_rewiring.py
-----------------
Graph rewiring based on Ollivier-Ricci curvature.

Background
----------
Ollivier-Ricci curvature (ORC) is a discrete analogue of Riemannian curvature
defined on graph edges. For an edge (u, v), ORC measures how efficiently
probability mass can be transported between the neighborhoods of u and v via
the Wasserstein-1 (Earth Mover's) distance.

    kappa(u, v) = 1 - W1(mu_u, mu_v) / d(u, v)

where mu_u and mu_v are uniform distributions over the neighbours of u and v,
and d(u, v) is the graph-geodesic distance between the two nodes.

Geometric intuition
-------------------
- kappa > 0  →  positive curvature  →  "fat" triangle, good information flow
- kappa < 0  →  negative curvature  →  "thin" bridge / bottleneck

Over-squashing mitigation
-------------------------
Edges with strongly negative curvature are structural bottlenecks: they are
the sole conduit for information between two otherwise poorly-connected
regions. By adding shortcut edges that bypass these bottlenecks (connecting
nodes in the neighbourhoods of the bottleneck's endpoints), we:

  1. Reduce the effective resistance / commute time across the cut.
  2. Increase the spectral gap of the graph Laplacian, which directly bounds
     the amount of information that can flow across the graph in k GNN layers.
  3. Provide alternative paths so that no single edge concentrates the entire
     information flow, thus relieving the exponential squashing.

The rewiring procedure follows the approach of Topping et al. (2022)
"Understanding over-squashing and bottlenecks on graphs via curvature."
"""

from __future__ import annotations

import itertools
from typing import Set, Tuple

import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx


def curvature_rewire(data: Data, k: int = 50) -> Data:
    """Rewire a graph by adding shortcut edges at low-curvature bottlenecks.

    The algorithm proceeds in five stages:

    1. **Convert** the PyG ``Data`` object to a NetworkX graph.
    2. **Compute** Ollivier-Ricci curvature for every edge using the
       ``GraphRicciCurvature`` library.
    3. **Select** the ``k`` edges with the most negative curvature — these
       are the structural bottlenecks most responsible for over-squashing.
    4. **Add shortcuts** by connecting pairs of neighbours from opposite
       endpoints of each bottleneck edge (i.e. for bottleneck (u, v), we
       consider edges (a, b) where a ∈ N(u) and b ∈ N(v)).
    5. **Convert** the augmented NetworkX graph back to a PyG ``Data``
       object, preserving all original node features and masks.

    Args:
        data: A PyTorch Geometric ``Data`` object. Must contain at minimum
            ``x``, ``edge_index``, ``y``, ``train_mask``, ``val_mask``,
            and ``test_mask``.
        k: Maximum number of new shortcut edges to add. Defaults to 50.
            The actual number added may be less if fewer unique shortcut
            candidates exist across all selected bottleneck edges.

    Returns:
        A new ``Data`` object with up to ``k`` additional edges.
        Node features, labels, and all masks are unchanged.

    Raises:
        ImportError: If the ``GraphRicciCurvature`` package is not installed.
            Install via: ``pip install GraphRicciCurvature``.

    Example:
        >>> data = load_cora()
        >>> data_rewired = curvature_rewire(data, k=100)
        >>> assert data_rewired.num_nodes == data.num_nodes  # nodes unchanged
    """
    try:
        from GraphRicciCurvature.OllivierRicci import OllivierRicci
    except ImportError as exc:
        raise ImportError(
            "GraphRicciCurvature is required for curvature rewiring. "
            "Install it with: pip install GraphRicciCurvature"
        ) from exc

    num_nodes: int = data.num_nodes

    # ------------------------------------------------------------------
    # 1. Convert PyG Data -> NetworkX undirected graph.
    #    We work in NetworkX throughout the curvature computation because
    #    OllivierRicci expects a NetworkX graph as input.
    # ------------------------------------------------------------------
    # to_networkx produces a DiGraph; convert to undirected for ORC.
    nx_graph: nx.Graph = to_networkx(data, to_undirected=True)
    density = nx.density(nx_graph)
    avg_degree = sum(dict(nx_graph.degree()).values()) / nx_graph.number_of_nodes()
    print(f"Density: {density:.4f}, Avg Degree: {avg_degree:.2f}")

    # Ensure the graph has exactly the expected number of nodes
    # (isolated nodes may be dropped by to_networkx in some versions).
    for node_id in range(num_nodes):
        if node_id not in nx_graph:
            nx_graph.add_node(node_id)

    # ------------------------------------------------------------------
    # 2. Compute Ollivier-Ricci curvature for all edges.
    #    OllivierRicci annotates each edge with a "ricciCurvature" attribute.
    # ------------------------------------------------------------------
    orc = OllivierRicci(nx_graph, alpha=0.5, verbose="ERROR")
    orc.compute_ricci_curvature()
    curved_graph: nx.Graph = orc.G.copy()  # graph with curvature edge attributes
    density = nx.density(curved_graph)
    avg_degree = sum(dict(curved_graph.degree()).values()) / curved_graph.number_of_nodes()
    print(f"Density: {density:.4f}, Avg Degree: {avg_degree:.2f}")

    # ------------------------------------------------------------------
    # 3. Sort edges by curvature (ascending) and select the most negative
    #    ones — these are the bottleneck edges we want to bypass.
    # ------------------------------------------------------------------
    edge_curvatures: list[Tuple[float, int, int]] = []

    for u, v, attrs in curved_graph.edges(data=True):
        curvature: float = attrs.get("ricciCurvature", 0.0)
        edge_curvatures.append((curvature, u, v))

    # Sort ascending: most negative curvature (worst bottlenecks) come first.
    edge_curvatures.sort(key=lambda t: t[0])

    # Take up to k candidate bottleneck edges to search for shortcuts.
    # We examine more candidates than k to maximise the chance of finding
    # k unique, non-duplicate shortcut edges.
    num_candidates: int = min(len(edge_curvatures), max(k, 10))
    bottleneck_edges: list[Tuple[int, int]] = [
        (u, v) for (_, u, v) in edge_curvatures[:num_candidates]
    ]

    # ------------------------------------------------------------------
    # 4. Generate shortcut edge candidates.
    #    For each bottleneck edge (u, v), we consider all pairs (a, b)
    #    where a ∈ N(u) \ {v} and b ∈ N(v) \ {u}, i.e. we connect
    #    neighbours on opposite sides of the bottleneck.
    #    Shortcuts are collected as unordered pairs to avoid duplicates.
    # ------------------------------------------------------------------
    # Collect all edges already in the graph as a set of canonical pairs.
    existing_edges: Set[Tuple[int, int]] = {
        (min(u, v), max(u, v))
        for u, v in curved_graph.edges()
    }

    new_edges: list[Tuple[int, int]] = []
    seen_candidates: Set[Tuple[int, int]] = set()

    for u, v in bottleneck_edges:
        if len(new_edges) >= k:
            break  # we have enough shortcut edges

        # Neighbours of u and v, excluding the bottleneck endpoints themselves.
        neighbors_u: list[int] = [n for n in curved_graph.neighbors(u) if n != v]
        neighbors_v: list[int] = [n for n in curved_graph.neighbors(v) if n != u]

        # Consider all cross-neighbourhood pairs as shortcut candidates.
        for a, b in itertools.product(neighbors_u, neighbors_v):
            if len(new_edges) >= k:
                break

            # Canonicalise the edge to avoid (a,b) / (b,a) duplicates.
            canonical: Tuple[int, int] = (min(a, b), max(a, b))

            # Skip self-loops, already-existing edges, and already-queued edges.
            if a == b:
                continue
            if canonical in existing_edges:
                continue
            if canonical in seen_candidates:
                continue

            seen_candidates.add(canonical)
            new_edges.append(canonical)

    # ------------------------------------------------------------------
    # 5. Add the shortcut edges to the NetworkX graph, then convert back
    #    to a PyG Data object and restore node features / masks.
    # ------------------------------------------------------------------
    curved_graph.add_edges_from(new_edges)  ## habra que ver en proteins como se hace

    # Build the new edge_index from the augmented NetworkX graph.
    # We reconstruct both directions for each undirected edge.
    all_edges: list[Tuple[int, int]] = list(curved_graph.edges())

    # Create directed edge list: add both (u->v) and (v->u) for each edge.
    src_list: list[int] = []
    dst_list: list[int] = []

    for u, v in all_edges:
        src_list.extend([u, v])
        dst_list.extend([v, u])

    new_edge_index = torch.tensor(
        [src_list, dst_list], dtype=torch.long
    )

    # Remove any duplicate directed edges that may arise from the conversion.
    new_edge_index = _deduplicate_edges(new_edge_index)

    # Assemble the final Data object, copying all original attributes.
    new_data = Data(
        x=data.x,
        edge_index=new_edge_index,
        y=data.y,
        train_mask=data.train_mask,
        val_mask=data.val_mask,
        test_mask=data.test_mask,
    )

    return new_data


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _deduplicate_edges(edge_index: torch.Tensor) -> torch.Tensor:
    """Remove duplicate directed edges from an edge_index tensor.

    Args:
        edge_index: A ``[2, E]`` tensor of directed edges, potentially
            containing duplicates.

    Returns:
        A ``[2, E']`` tensor with all duplicate edges removed (E' <= E).
    """
    # Encode each edge as a single scalar: src * (max_node+1) + dst,
    # then use torch.unique to deduplicate.
    num_nodes: int = int(edge_index.max().item()) + 1
    edge_codes = edge_index[0] * num_nodes + edge_index[1]
    unique_codes = torch.unique(edge_codes, sorted=False, return_inverse=False)
    print(len(unique_codes))
    print(unique_codes)

    src = unique_codes // num_nodes
    dst = unique_codes % num_nodes

    return torch.stack([src, dst], dim=0)

"""
ricci_rewiring.py
-----------------
General-purpose graph rewiring based on Ollivier-Ricci curvature.

Supports both single-graph datasets (e.g. Cora, PubMed) and multi-graph
datasets (e.g. PROTEINS, ENZYMES, ZINC). Designed for safe use inside
training pipelines: the original Data object is never modified in-place.

Background
----------
Ollivier-Ricci curvature (ORC) is a discrete analogue of Riemannian curvature
defined on graph edges. For an edge (u, v), ORC measures how efficiently
probability mass can be transported between the neighbourhoods of u and v via
the Wasserstein-1 (Earth Mover's) distance:

    kappa(u, v) = 1 - W1(mu_u, mu_v) / d(u, v)

where mu_u, mu_v are uniform distributions over the neighbours of u and v,
and d(u, v) is the shortest-path distance.

Geometric intuition
-------------------
- kappa > 0  ->  positive curvature  ->  well-connected region, good message flow
- kappa < 0  ->  negative curvature  ->  bottleneck / bridge, poor message flow

Over-squashing
--------------
Edges with strongly negative curvature force exponentially many messages
through a single channel, squashing information irreversibly. Adding shortcut
edges at these bottlenecks increases the spectral gap of the graph Laplacian,
providing alternative paths and reducing effective resistance.

Reference: Topping et al. (2022) "Understanding over-squashing and bottlenecks
on graphs via curvature." ICLR 2022.
"""

from __future__ import annotations

import itertools
import logging
from typing import Dict, List, Optional, Set, Tuple, Union

import networkx as nx
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import to_networkx

# logger = logging.getLogger("experiment")

import os

def _build_file_logger(log_path: str):
    logger = logging.getLogger(f"ricci_rewire_{log_path}")
    logger.setLevel(logging.INFO)

    # evitar duplicados
    if not logger.handlers:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        handler = logging.FileHandler(log_path)
        handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s"
        )
        handler.setFormatter(formatter)

        logger.addHandler(handler)

    return logger


# =============================================================================
# Public API
# =============================================================================

def curvature_rewire(
    data: Union[Data, List[Data]],
    k: int = 50,
    alpha: float = 0.5,
    approximate: bool = False,
    log_path: Optional[str] = None,
) -> Union[Data, List[Data]]:
    """Rewire a graph (or list of graphs) by adding shortcuts at curvature bottlenecks.

    Identifies edges with the most negative Ollivier-Ricci curvature and adds
    up to ``k`` shortcut edges per graph by connecting neighbours on opposite
    sides of each bottleneck. All existing node/edge attributes are preserved.

    Args:
        data: Either a single PyG ``Data`` object or a list of ``Data`` objects.
            Masks (``train_mask``, ``val_mask``, ``test_mask``) are preserved
            when present but are not required.
        k: Maximum number of new shortcut edges to add per graph. The actual
            number may be lower if the graph has insufficient bottleneck
            neighbourhood pairs. Defaults to 50.
        alpha: Laziness parameter for Ollivier-Ricci curvature in [0, 1].
            Higher values place more probability mass on the node itself
            (lazy random walk). Defaults to 0.5.
        approximate: If ``True``, replace exact ORC computation with an
            edge-betweenness centrality proxy. Substantially faster on large
            graphs, at the cost of approximation quality. Defaults to ``False``.

    Returns:
        A rewired ``Data`` object if the input was a single ``Data``, or a
        list of rewired ``Data`` objects if the input was a list. The original
        input is never modified.

    Raises:
        TypeError: If ``data`` is neither a ``Data`` nor a list of ``Data``.
        ImportError: If ``GraphRicciCurvature`` is not installed and
            ``approximate=False``.

    Example -- single graph (node classification)::

        data = load_cora()
        data_rewired = curvature_rewire(data, k=50)

    Example -- dataset (graph classification)::

        dataset = TUDataset(root="/tmp", name="PROTEINS")
        rewired = curvature_rewire(list(dataset), k=30, approximate=True)
    """
    if log_path is not None:
        local_logger = _build_file_logger(log_path)
    else:
        local_logger = logging.getLogger("experiment")

    if isinstance(data, Data):
        return _rewire_single_graph(data, k=k, alpha=alpha, approximate=approximate, logger=local_logger)

    if isinstance(data, Dataset):
        return [
            _rewire_single_graph(data[i], k=k, alpha=alpha, approximate=approximate, logger=local_logger)
            for i in range(len(data))
        ]

    if isinstance(data, list):
        return [
            _rewire_single_graph(g, k=k, alpha=alpha, approximate=approximate, logger=local_logger)
            for g in data
        ]

    raise TypeError(
        f"curvature_rewire expects Data, list[Data], or Dataset, "
        f"got {type(data).__name__}."
    )


# =============================================================================
# Core per-graph logic
# =============================================================================

def _rewire_single_graph(
    data: Data,
    k: int,
    alpha: float,
    approximate: bool,
    logger,
) -> Data:
    """Apply curvature rewiring to a single PyG Data object.

    This is the main per-graph routine. It converts to NetworkX, computes
    curvature, identifies bottlenecks, generates shortcut candidates, and
    reconstructs a new Data object with the augmented edge_index.

    Args:
        data: A single PyG ``Data`` object.
        k: Maximum number of shortcut edges to add.
        alpha: ORC laziness parameter (ignored when ``approximate=True``).
        approximate: Use edge-betweenness as a fast curvature proxy.

    Returns:
        A new ``Data`` object with up to ``k`` additional edges.
    """
    num_nodes: int = data.num_nodes
    num_edges = data.edge_index.size(1) // 2  # undirected

    # override k dinámicamente
    k = max(5, int(0.1 * num_edges))

    # logger.info(f"[Ricci] num_edges={num_edges}, using k={k}")

    # ------------------------------------------------------------------
    # Step 1 -- Convert PyG -> NetworkX undirected graph.
    #           OllivierRicci operates on NetworkX graphs.
    # ------------------------------------------------------------------
    nx_graph: nx.Graph = _pyg_to_networkx(data, num_nodes)

    edges_before: int = nx_graph.number_of_edges()
    density_before: float = nx.density(nx_graph)
    avg_degree_before: float = (
        sum(d for _, d in nx_graph.degree()) / max(num_nodes, 1)
    )

    # ------------------------------------------------------------------
    # Step 2 -- Compute edge curvature (exact ORC or fast approximation).
    # ------------------------------------------------------------------
    curvature_map: Dict[Tuple[int, int], float] = _compute_curvature(
        nx_graph, alpha=alpha, approximate=approximate
    )

    # ------------------------------------------------------------------
    # Step 3 -- Rank edges by curvature (ascending); pick worst bottlenecks.
    #           We examine a window of candidates to find k unique shortcuts.
    # ------------------------------------------------------------------
    bottleneck_edges: List[Tuple[int, int]] = _rank_by_curvature(curvature_map, k)

    # ------------------------------------------------------------------
    # Step 4 -- Generate shortcut edges from cross-neighbourhood pairs.
    # ------------------------------------------------------------------
    existing_edges: Set[Tuple[int, int]] = {
        (min(u, v), max(u, v)) for u, v in nx_graph.edges()
    }

    new_edges: List[Tuple[int, int]] = _generate_shortcuts(
        bottleneck_edges=bottleneck_edges,
        nx_graph=nx_graph,
        existing_edges=existing_edges,
        k=k,
    )

    # ------------------------------------------------------------------
    # Step 5 -- Build augmented edge_index and assemble the new Data object.
    # ------------------------------------------------------------------
    new_edge_index: torch.Tensor = _build_edge_index(
        new_edges=new_edges,
        existing_edge_index=data.edge_index,
    )

    new_data: Data = _copy_data_with_new_edges(data, new_edge_index, logger)

    # ------------------------------------------------------------------
    # Logging -- report structural changes for research transparency.
    # ------------------------------------------------------------------
    edges_after: int = new_edge_index.size(1) // 2  # approximate undirected count
    density_after: float = (
        2 * edges_after / max(num_nodes * (num_nodes - 1), 1)
    )
    avg_degree_after: float = (2 * edges_after) / max(num_nodes, 1)

    logger.info(
        "Rewiring complete | nodes=%d | edges %d->%d (+%d) | "
        "density %.4f->%.4f | avg_degree %.2f->%.2f",
        num_nodes,
        edges_before, edges_after, len(new_edges),
        density_before, density_after,
        avg_degree_before, avg_degree_after,
    )

    return new_data


# =============================================================================
# Curvature computation
# =============================================================================

def _compute_curvature(
    nx_graph: nx.Graph,
    alpha: float,
    approximate: bool,
) -> Dict[Tuple[int, int], float]:
    """Compute a curvature score for every edge in the graph.

    Two modes are supported:

    **Exact (default):** Uses Ollivier-Ricci curvature via the
    ``GraphRicciCurvature`` library. Accurate but O(|E| * |V|^2) in the
    worst case.

    **Approximate:** Uses negative edge-betweenness centrality as a fast
    proxy. Edges with high betweenness are structural bridges (analogous to
    negative curvature). Runs in O(|V| * |E|) -- practical for large graphs.

    Args:
        nx_graph: An undirected NetworkX graph.
        alpha: Laziness parameter for exact ORC (ignored if ``approximate``).
        approximate: If ``True``, use edge-betweenness as the proxy.

    Returns:
        A dict mapping each canonical edge ``(min(u,v), max(u,v))`` to its
        curvature score. Lower (more negative) scores indicate bottlenecks.

    Raises:
        ImportError: If ``approximate=False`` and ``GraphRicciCurvature``
            is not installed.
    """
    if approximate:
        return _betweenness_proxy(nx_graph)

    # --- Exact Ollivier-Ricci curvature ---
    try:
        from GraphRicciCurvature.OllivierRicci import OllivierRicci
    except ImportError as exc:
        raise ImportError(
            "GraphRicciCurvature is required for exact curvature rewiring.\n"
            "Install it with:  pip install GraphRicciCurvature\n"
            "Or use approximate=True for a fast edge-betweenness proxy."
        ) from exc

    orc = OllivierRicci(nx_graph, alpha=alpha, verbose="ERROR")
    orc.compute_ricci_curvature()

    curvature_map: Dict[Tuple[int, int], float] = {}
    for u, v, attrs in orc.G.edges(data=True):
        canonical = (min(u, v), max(u, v))
        curvature_map[canonical] = float(attrs.get("ricciCurvature", 0.0))

    return curvature_map


def _betweenness_proxy(
    nx_graph: nx.Graph,
) -> Dict[Tuple[int, int], float]:
    """Compute negative edge-betweenness centrality as a curvature proxy.

    Edges that lie on many shortest paths (high betweenness) behave as
    structural bottlenecks, analogously to edges with negative Ricci
    curvature. We negate the score so that the convention
    "lower value = worse bottleneck" is preserved, matching the ORC interface.

    Args:
        nx_graph: An undirected NetworkX graph.

    Returns:
        A dict mapping each canonical edge ``(min(u,v), max(u,v))`` to its
        negated betweenness score.
    """
    raw: Dict[Tuple[int, int], float] = nx.edge_betweenness_centrality(
        nx_graph, normalized=True
    )
    # Negate: high betweenness -> low (most negative) score -> selected first.
    return {
        (min(u, v), max(u, v)): -score
        for (u, v), score in raw.items()
    }


# =============================================================================
# Bottleneck selection and shortcut generation
# =============================================================================

def _rank_by_curvature(
    curvature_map: Dict[Tuple[int, int], float],
    k: int,
) -> List[Tuple[int, int]]:
    """Sort edges by curvature ascending and return the worst bottlenecks.

    We examine a window larger than ``k`` to increase the probability of
    finding ``k`` distinct, non-duplicate shortcut edges across all
    bottleneck neighbourhoods.

    Args:
        curvature_map: Mapping from canonical edge to curvature score.
        k: Desired number of shortcut edges.

    Returns:
        An ordered list of ``(u, v)`` bottleneck edges, most negative first.
    """
    sorted_items = sorted(curvature_map.items(), key=lambda kv: kv[1])

    # Examine at least k candidates (or all edges if the graph is small).
    window: int = min(len(sorted_items), max(k, 10))
    return [edge for edge, _ in sorted_items[:window]]


def _generate_shortcuts(
    bottleneck_edges: List[Tuple[int, int]],
    nx_graph: nx.Graph,
    existing_edges: Set[Tuple[int, int]],
    k: int,
) -> List[Tuple[int, int]]:
    """Generate up to k shortcut edges from bottleneck neighbourhoods.

    For each bottleneck edge (u, v), considers all pairs (a, b) where
    a in N(u) \\ {v} and b in N(v) \\ {u}. These cross-neighbourhood pairs
    provide alternative paths that bypass the bottleneck.

    Args:
        bottleneck_edges: Ranked list of ``(u, v)`` bottleneck edges.
        nx_graph: The original undirected NetworkX graph.
        existing_edges: Set of canonical edges already in the graph.
        k: Maximum number of shortcuts to return.

    Returns:
        A list of up to ``k`` new canonical edges ``(min(a,b), max(a,b))``.
    """
    new_edges: List[Tuple[int, int]] = []
    # Track queued candidates to prevent duplicates across bottleneck iterations.
    seen: Set[Tuple[int, int]] = set()

    for u, v in bottleneck_edges:
        if len(new_edges) >= k:
            break

        # One-hop neighbourhoods, excluding the bottleneck endpoints themselves.
        neighbors_u: List[int] = [n for n in nx_graph.neighbors(u) if n != v]
        neighbors_v: List[int] = [n for n in nx_graph.neighbors(v) if n != u]

        # Cross-product of both neighbourhoods gives shortcut candidates.
        for a, b in itertools.product(neighbors_u, neighbors_v):
            if len(new_edges) >= k:
                break

            if a == b:
                # Discard self-loops.
                continue

            canonical: Tuple[int, int] = (min(a, b), max(a, b))

            if canonical in existing_edges or canonical in seen:
                # Discard edges already in the graph or already queued.
                continue

            seen.add(canonical)
            new_edges.append(canonical)

    return new_edges


# =============================================================================
# Edge index construction
# =============================================================================

def _build_edge_index(
    new_edges: List[Tuple[int, int]],
    existing_edge_index: torch.Tensor,
) -> torch.Tensor:
    """Construct a deduplicated directed edge_index from original + shortcut edges.

    Starts from the original PyG ``edge_index`` (which already encodes both
    directions of each undirected edge) and appends the new shortcut edges
    in both directions before deduplicating.

    Args:
        new_edges: List of new canonical undirected edges to add.
        existing_edge_index: The original ``[2, E]`` PyG edge_index tensor.

    Returns:
        A deduplicated ``[2, E']`` directed edge_index tensor.
    """
    if not new_edges:
        return existing_edge_index.clone()

    # Expand each undirected shortcut into two directed edges.
    src_new: List[int] = []
    dst_new: List[int] = []

    for u, v in new_edges:
        src_new.extend([u, v])
        dst_new.extend([v, u])

    shortcut_index = torch.tensor([src_new, dst_new], dtype=torch.long)

    # Concatenate with the original edge_index and remove duplicates.
    combined = torch.cat([existing_edge_index, shortcut_index], dim=1)
    return _deduplicate_edges(combined)


def _deduplicate_edges(edge_index: torch.Tensor) -> torch.Tensor:
    """Remove duplicate directed edges from an edge_index tensor.

    Encodes each directed edge ``(u, v)`` as a single integer
    ``u * num_nodes + v``, applies ``torch.unique``, then decodes back.
    This is O(E log E) and avoids any Python-level loops.

    Args:
        edge_index: A ``[2, E]`` tensor of directed edges, possibly containing
            duplicates.

    Returns:
        A ``[2, E']`` tensor with all duplicates removed (E' <= E).
    """
    num_nodes: int = int(edge_index.max().item()) + 1
    # Each edge (u, v) -> unique scalar in [0, num_nodes^2).
    edge_codes = edge_index[0] * num_nodes + edge_index[1]
    # torch.unique with no extra flags returns a single 1-D tensor.
    unique_codes = torch.unique(edge_codes, sorted=False)

    src = unique_codes // num_nodes
    dst = unique_codes % num_nodes

    return torch.stack([src, dst], dim=0)


# =============================================================================
# Data object utilities
# =============================================================================

def _pyg_to_networkx(data: Data, num_nodes: int) -> nx.Graph:
    """Convert a PyG Data object to an undirected NetworkX graph.

    Ensures all node indices [0, num_nodes) are present, even if some are
    isolated (``to_networkx`` may drop them in certain PyG versions).

    Args:
        data: A PyG ``Data`` object with an ``edge_index`` attribute.
        num_nodes: Total number of nodes (used to add any missing isolates).

    Returns:
        An undirected ``nx.Graph`` with nodes labelled 0 ... num_nodes-1.
    """
    nx_graph: nx.Graph = to_networkx(data, to_undirected=True)

    # Guarantee all node indices are present — isolated nodes may be dropped.
    for node_id in range(num_nodes):
        if node_id not in nx_graph:
            nx_graph.add_node(node_id)

    return nx_graph


def _copy_data_with_new_edges(
    data: Data,
    new_edge_index: torch.Tensor,
    logger,
) -> Data:
    """Create a new Data object identical to ``data`` but with a new edge_index.

    All attributes stored on the original object (``x``, ``y``, masks,
    ``batch``, etc.) are shallow-copied to the new object. ``edge_attr`` is
    dropped with a warning if the edge count changes. The original object is
    never modified.

    Args:
        data: The original PyG ``Data`` object.
        new_edge_index: The replacement ``[2, E']`` edge_index tensor.

    Returns:
        A new ``Data`` object sharing all attributes of ``data`` except
        ``edge_index``, which is replaced by ``new_edge_index``.
    """
    new_data = Data()

    for key, value in data:
        if key == "edge_index":
            continue  # replaced below

        if key == "edge_attr":
            # Edge attributes are tied to specific edges; they cannot be
            # transferred when the edge set changes size or order.
            if data.edge_index.size(1) != new_edge_index.size(1):
                logger.warning(
                    "edge_attr dropped during rewiring: original edge count %d "
                    "differs from rewired edge count %d.",
                    data.edge_index.size(1),
                    new_edge_index.size(1),
                )
                continue

        setattr(new_data, key, value)

    new_data.edge_index = new_edge_index
    return new_data

"""
ricci_rewiring.py
-----------------
Graph rewiring strategy: Ollivier-Ricci Curvature Rewiring.

Ollivier-Ricci curvature is a discrete analogue of Riemannian curvature
defined on the edges of a graph.  An edge (u, v) has *negative* curvature when
the neighbourhoods of u and v are poorly connected to each other — exactly the
structural signature of a bottleneck that causes over-squashing.

This implementation computes curvature directly using scipy (earth-mover /
Wasserstein-1 distance via linear programming), with NO multiprocessing and
NO external GraphRicciCurvature dependency.  It runs fully in the main thread
so progress can be tracked edge-by-edge with tqdm.

Dependencies
------------
    pip install networkx scipy torch_geometric tqdm
"""

import random
from typing import Dict, List, Set, Tuple

import networkx as nx
import numpy as np
import torch
from scipy.optimize import linprog
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Ollivier-Ricci curvature — pure Python / scipy, no multiprocessing
# ---------------------------------------------------------------------------

def _lazy_distribution(
    G: nx.Graph,
    node: int,
    alpha: float = 0.5,
) -> Dict[int, float]:
    """Build the lazy random-walk distribution centred at a node.

    Places mass ``alpha`` on the node itself and distributes the remaining
    ``1 - alpha`` uniformly over its neighbours.

    Args:
        G: The undirected graph.
        node: The node at which to centre the distribution.
        alpha: Laziness parameter in [0, 1].  Defaults to 0.5.

    Returns:
        A dict mapping node ids to probability masses summing to 1.
    """
    neighbors = list(G.neighbors(node))
    dist: Dict[int, float] = {node: alpha}

    if neighbors:
        nbr_mass = (1.0 - alpha) / len(neighbors)
        for nbr in neighbors:
            dist[nbr] = dist.get(nbr, 0.0) + nbr_mass

    return dist


def _earth_mover_distance(
    mu: Dict[int, float],
    nu: Dict[int, float],
    shortest_paths: Dict[int, Dict[int, float]],
) -> float:
    """Compute the Wasserstein-1 distance between two distributions via LP.

    Solves the discrete optimal-transport programme:

        min   sum_{i,j} d(i,j) * T[i,j]
        s.t.  sum_j T[i,j] = mu[i]   (supply)
              sum_i T[i,j] = nu[j]   (demand)
              T[i,j] >= 0

    Args:
        mu: Source distribution {node: probability}.
        nu: Target distribution {node: probability}.
        shortest_paths: Pre-computed all-pairs shortest-path lengths.

    Returns:
        W1(mu, nu) as a float.
    """
    src_nodes = list(mu.keys())
    tgt_nodes = list(nu.keys())
    n, m = len(src_nodes), len(tgt_nodes)

    # Flatten the n×m cost matrix into the LP objective vector.
    c = np.array(
        [shortest_paths[i].get(j, 1.0) for i in src_nodes for j in tgt_nodes],
        dtype=float,
    )

    # Equality constraints matrix: n supply rows + m demand rows.
    A_eq = np.zeros((n + m, n * m), dtype=float)
    b_eq = np.zeros(n + m, dtype=float)

    for i, node_i in enumerate(src_nodes):          # supply
        A_eq[i, i * m : (i + 1) * m] = 1.0
        b_eq[i] = mu[node_i]

    for j in range(m):                               # demand
        A_eq[n + j, j::m] = 1.0
        b_eq[n + j] = nu[tgt_nodes[j]]

    result = linprog(c, A_eq=A_eq, b_eq=b_eq,
                     bounds=[(0, None)] * (n * m), method="highs")

    return float(result.fun) if result.success else 1.0


def _ollivier_ricci_edge(
    G: nx.Graph,
    u: int,
    v: int,
    shortest_paths: Dict[int, Dict[int, float]],
    alpha: float = 0.5,
) -> float:
    """Compute Ollivier-Ricci curvature for a single edge (u, v).

    κ(u, v) = 1 - W₁(μ_u, μ_v)

    (d(u,v) = 1 for all edges in an unweighted graph, so the denominator
    is always 1 and is omitted.)

    Args:
        G: The undirected graph.
        u: Source node of the edge.
        v: Target node of the edge.
        shortest_paths: Pre-computed all-pairs shortest-path lengths.
        alpha: Laziness parameter.  Defaults to 0.5.

    Returns:
        Scalar curvature κ(u, v).
    """
    mu = _lazy_distribution(G, u, alpha)
    nu = _lazy_distribution(G, v, alpha)
    w1 = _earth_mover_distance(mu, nu, shortest_paths)
    return 1.0 - w1


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_edge_set(edge_index: torch.Tensor) -> Set[Tuple[int, int]]:
    """Convert a COO edge index into a set of (src, dst) tuples for O(1) look-ups.

    Args:
        edge_index: Long tensor of shape ``[2, E]``.

    Returns:
        A set of ``(src, dst)`` integer pairs covering every directed edge.
    """
    return set(zip(edge_index[0].tolist(), edge_index[1].tolist()))


def _candidate_shortcuts(
    G: nx.Graph,
    u: int,
    v: int,
    existing_edges: Set[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    """Generate candidate shortcut edges for a bottleneck edge (u, v).

    Candidates are cross-neighbourhood pairs (a, b) where a ∈ N(u) \ {v}
    and b ∈ N(v) \ {u}.  These bypass the bottleneck without duplicating
    any existing edge.

    Args:
        G: The undirected graph.
        u: Source node of the bottleneck edge.
        v: Target node of the bottleneck edge.
        existing_edges: Set of directed edges already in the graph.

    Returns:
        List of ``(a, b)`` pairs safe to add as new edges.
    """
    u_nbrs = [n for n in G.neighbors(u) if n != v]
    v_nbrs = [n for n in G.neighbors(v) if n != u]

    candidates: List[Tuple[int, int]] = []
    for a in u_nbrs:
        for b in v_nbrs:
            if a != b and (a, b) not in existing_edges and (b, a) not in existing_edges:
                candidates.append((a, b))
    return candidates


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def curvature_rewire(data: Data, k: int = 50, alpha: float = 0.5) -> Data:
    """Rewire a graph by adding shortcut edges at Ollivier-Ricci bottlenecks.

    Uses a pure Python / scipy implementation — no GraphRicciCurvature
    library, no multiprocessing, works on all platforms including Windows.
    Progress is reported edge-by-edge via tqdm.

    Algorithm overview
    ------------------
    1. Convert PyG ``Data`` → undirected NetworkX graph.
    2. Pre-compute all-pairs shortest-path lengths (fast BFS, done once).
    3. Compute κ(u, v) for every edge in a single-threaded loop with tqdm.
    4. Sort edges by κ ascending (most negative = biggest bottleneck).
    5. Add up to ``k`` shortcut edges from bottleneck neighbourhoods.
    6. Return a new ``Data`` object with the augmented ``edge_index``.

    Args:
        data: A PyTorch Geometric ``Data`` object with attributes:
            ``x``, ``edge_index``, ``y``, ``train_mask``, ``val_mask``,
            ``test_mask``.
        k: Maximum number of new undirected shortcut edges to add.
            Each becomes two directed edges in ``edge_index``.
            Defaults to ``50``.
        alpha: Laziness of the random-walk distribution.  Defaults to ``0.5``.

    Returns:
        A new ``Data`` object with up to ``2*k`` additional directed edges.
    """
    # ------------------------------------------------------------------
    # 1. Convert to undirected NetworkX graph.
    # ------------------------------------------------------------------
    G: nx.Graph = to_networkx(data, to_undirected=True)
    tqdm.write(
        f"[Ricci rewiring] {G.number_of_nodes()} nodes, "
        f"{G.number_of_edges()} edges."
    )

    # ------------------------------------------------------------------
    # 2. Pre-compute all-pairs shortest paths (BFS, O(N*(N+E))).
    #    For Cora (2 708 nodes, 5 278 edges) this takes ~1 s.
    # ------------------------------------------------------------------
    tqdm.write("[Ricci rewiring] Pre-computing shortest paths (BFS)...")
    shortest_paths: Dict[int, Dict[int, float]] = dict(
        nx.all_pairs_shortest_path_length(G)
    )

    # ------------------------------------------------------------------
    # 3. Compute curvature edge-by-edge in the main thread.
    #    Each call solves a small LP (~10-30 variables) via HiGHS.
    #    Cora (~5 278 edges) typically finishes in 30-90 s on CPU.
    # ------------------------------------------------------------------
    edges = list(G.edges())
    curvature_values: List[Tuple[float, int, int]] = []

    for u, v in tqdm(edges, desc="[1/2] Ricci curvature",
                     unit="edge", colour="cyan", dynamic_ncols=True):
        kappa = _ollivier_ricci_edge(G, u, v, shortest_paths, alpha)
        curvature_values.append((kappa, u, v))

    # Sort: lowest (most negative) curvature first.
    curvature_values.sort(key=lambda t: t[0])

    # ------------------------------------------------------------------
    # 4. Build existing edge set for fast duplicate checking.
    # ------------------------------------------------------------------
    existing_edges: Set[Tuple[int, int]] = _build_edge_set(data.edge_index)

    # ------------------------------------------------------------------
    # 5. Collect up to k shortcut edges.
    # ------------------------------------------------------------------
    new_edges: List[Tuple[int, int]] = []

    with tqdm(total=k, desc="[2/2] Shortcut edges",
              unit="edge", colour="green", dynamic_ncols=True) as pbar:

        for _kappa, u, v in curvature_values:
            if len(new_edges) >= k:
                break

            candidates = _candidate_shortcuts(G, u, v, existing_edges)
            random.Random(42).shuffle(candidates)

            for a, b in candidates:
                if len(new_edges) >= k:
                    break
                new_edges.append((a, b))
                existing_edges.add((a, b))
                existing_edges.add((b, a))
                pbar.update(1)

    tqdm.write(f"[Ricci rewiring] Done — added {len(new_edges)} shortcut edges.")

    # ------------------------------------------------------------------
    # 6. Append shortcuts (bidirectional) to edge_index.
    # ------------------------------------------------------------------
    if new_edges:
        src = [a for a, b in new_edges] + [b for a, b in new_edges]
        dst = [b for a, b in new_edges] + [a for a, b in new_edges]
        shortcut_index = torch.tensor([src, dst], dtype=torch.long)
        edge_index_new = torch.cat([data.edge_index, shortcut_index], dim=1)
    else:
        edge_index_new = data.edge_index

    # ------------------------------------------------------------------
    # 7. Assemble and return rewired Data object.
    # ------------------------------------------------------------------
    return Data(
        x          = data.x,
        edge_index = edge_index_new,
        y          = data.y,
        train_mask = data.train_mask,
        val_mask   = data.val_mask,
        test_mask  = data.test_mask,
    )


# ---------------------------------------------------------------------------
# Algorithm explanation
# ---------------------------------------------------------------------------
"""
HOW RICCI CURVATURE REWIRING WORKS
===================================
Ollivier-Ricci curvature κ(u, v) measures how "close" the probability
distributions of a random walk starting at u and at v are, in the
Wasserstein-1 (earth-mover) sense:

    κ(u, v) = 1 - W₁(μ_u, μ_v)

where μ_u places mass α on u and distributes the rest uniformly over its
neighbours.  W₁ is computed by solving a small optimal-transport LP.

Interpretation:
    κ > 0  →  positively curved: neighbourhoods overlap (triangle-rich).
    κ = 0  →  flat: tree-like local structure.
    κ < 0  →  negatively curved: bottleneck — the two sides are far apart.

WHY IT HELPS OVER-SQUASHING
=============================
Over-squashing occurs when long-range paths are forced through a narrow
edge, compressing exponentially many messages into a fixed-size vector.
Negative Ricci curvature identifies exactly those edges.  Adding shortcut
edges in their neighbourhoods creates alternative paths, reduces effective
resistance between distant nodes, and allows long-range signals to
propagate in fewer hops — without needing deeper, over-smoothing GNNs.

References
----------
- Ollivier, Y. (2009). Ricci curvature of Markov chains on metric spaces.
- Topping et al. (2022). Understanding over-squashing via curvature. ICLR.
- Di Giovanni et al. (2023). On over-squashing in MPNNs. ICML.
"""
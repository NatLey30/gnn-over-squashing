# Over-Squashing in Graph Neural Networks

## Research question
Does graph rewiring mitigate over-squashing in message passing GNNs?

## Dataset
Cora citation network

## Models
- GCN
- GraphSAGE
- GAT

## Rewiring methods
- Virtual nodes
- Curvature-based rewiring

## Reproducing results

python main.py experiments/baseline_gcn.yaml
python main.py experiments/baseline_graphsage.yaml
python main.py experiments/baseline_gat.yaml

python main.py experiments/virtual_node_gcn.yaml
python main.py experiments/virtual_node_graphsage.yaml
python main.py experiments/virtual_node_gat.yaml

python main.py experiments/rewiring_curvature_gcn.yaml
python main.py experiments/rewiring_curvature_graphsage.yaml
python main.py experiments/rewiring_curvature_gat.yaml
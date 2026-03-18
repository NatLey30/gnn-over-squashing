import torch
from src.utils.metrics import accuracy


def evaluate_node_classification(model, data):

    model.eval()

    with torch.no_grad():
        out = model(data.x, data.edge_index)

    acc = accuracy(out[data.test_mask], data.y[data.test_mask])

    return acc


def evaluate_graph_classification(model, loader, device):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for data in loader:

            data = data.to(device)

            out = model(data.x, data.edge_index, data.batch)

            pred = out.argmax(dim=1)

            correct += (pred == data.y).sum().item()
            total += data.y.size(0)

    return correct / total

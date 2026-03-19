import torch
from src.utils.training import accuracy, get_masks


def evaluate_node_classification(model, data, split=0):

    model.eval()

    _, _, test_mask = get_masks(data, split)

    with torch.no_grad():
        out = model(data.x, data.edge_index)

    acc = accuracy(out[test_mask], data.y[test_mask])

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

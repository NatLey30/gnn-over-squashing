import torch
import torch.nn.functional as F
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


def evaluate_graph_regression(model, loader, device):

    model.eval()
    total_error = 0

    with torch.no_grad():
        for data in loader:
            data = data.to(device)

            out = model(data.x, data.edge_index, data.batch)
            # y = data.y.float()
            y = data.y.float().view(-1, 1) if data.y.dim() == 1 else data.y.float()

            error = F.l1_loss(out, y, reduction="mean")
            total_error += error.item()

    mae = total_error / len(loader.dataset)

    return mae

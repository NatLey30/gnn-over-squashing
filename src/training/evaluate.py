import torch
import torch.nn.functional as F
from src.utils.training import accuracy, get_masks, compute_loss, auc, hits_at_k


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


def evaluate_graph_regression(model, loader, device, mean, std):

    model.eval()
    total_error = 0

    with torch.no_grad():
        for data in loader:
            data = data.to(device)

            out = model(data.x, data.edge_index, data.batch)
            # y = data.y.view(-1, 1).float()
            y = (data.y.view(-1, 1).float() - mean) / std

            error = F.l1_loss(out, y, reduction="sum")  # MAE total
            total_error += error.item()

    mae = total_error / len(loader.dataset)

    return mae


def evaluate_link_prediction(model, data, split_edge):

    model.eval()

    with torch.no_grad():
        z = model(data.x, data.edge_index)

        # ===== TEST =====
        pos_edge_index = split_edge["test"]["edge"].t()
        neg_edge_index = split_edge["test"]["edge_neg"].t()

        _, test_pred, test_label = compute_loss(
            z,
            pos_edge_index,
            neg_edge_index
        )

        test_auc = auc(test_pred, test_label)
        test_hits = hits_at_k(
                test_pred[:len(pos_edge_index[0])],
                test_pred[len(pos_edge_index[0]):]
            )

    return test_auc, test_hits

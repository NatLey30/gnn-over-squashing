import torch
import torch.nn.functional as F

import copy
from tqdm import tqdm

from src.utils.training import accuracy, get_masks


def train_node_classification(model, data, optimizer, epochs=200, split=0, patience=10):

    train_mask, val_mask, _ = get_masks(data, split)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    # best_val_loss = float("inf")
    # # best_model_state = None
    # patience_counter = 0

    for epoch in tqdm(range(epochs)):

        # ===== TRAIN =====
        model.train()
        optimizer.zero_grad()

        out = model(data.x, data.edge_index)

        loss = F.cross_entropy(
            out[train_mask],
            data.y[train_mask]
        )

        loss.backward()
        optimizer.step()

        train_acc = accuracy(out[train_mask], data.y[train_mask])

        # ===== VALIDATION =====
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)

            val_loss = F.cross_entropy(
                out[val_mask],
                data.y[val_mask]
            )

            val_acc = accuracy(out[val_mask], data.y[val_mask])

         # ===== EARLY STOPPING =====
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     # best_model_state = copy.deepcopy(model.state_dict())
        #     patience_counter = 0
        # else:
        #     patience_counter += 1

        # if patience_counter >= patience:
        #     print(f"Early stopping at epoch {epoch}")
        #     break

        # ===== LOG =====
        history["train_loss"].append(loss.item())
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss.item())
        history["val_acc"].append(val_acc)

    return history


# def train_graph_classification(model, loader, optimizer, device, epochs=200):

#     model.train()

#     history = {
#         "train_loss": [],
#         "train_acc": []
#     }

#     for epoch in tqdm(range(epochs)):

#         total_loss = 0
#         correct = 0
#         total = 0

#         for data in loader:

#             data = data.to(device)

#             optimizer.zero_grad()

#             out = model(data.x, data.edge_index, data.batch)

#             loss = F.cross_entropy(out, data.y)

#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()

#             pred = out.argmax(dim=1)

#             correct += (pred == data.y).sum().item()
#             total += data.y.size(0)

#         acc = correct / total

#         history["train_loss"].append(total_loss / len(loader))
#         history["train_acc"].append(acc)

#     return history


def train_graph_classification(model, train_loader, val_loader, optimizer, device, epochs=200):

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    for epoch in tqdm(range(epochs)):

        # ===== TRAIN =====
        model.train()

        total_loss = 0
        correct = 0
        total = 0

        for data in train_loader:

            data = data.to(device)

            optimizer.zero_grad()

            out = model(data.x, data.edge_index, data.batch)

            loss = F.cross_entropy(out, data.y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            pred = out.argmax(dim=1)
            correct += (pred == data.y).sum().item()
            total += data.y.size(0)

        train_acc = correct / total
        train_loss = total_loss / len(train_loader)

        # ===== VALIDATION =====
        model.eval()

        val_loss_total = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for data in val_loader:

                data = data.to(device)

                out = model(data.x, data.edge_index, data.batch)

                loss = F.cross_entropy(out, data.y)

                val_loss_total += loss.item()

                pred = out.argmax(dim=1)
                val_correct += (pred == data.y).sum().item()
                val_total += data.y.size(0)

        val_acc = val_correct / val_total
        val_loss = val_loss_total / len(val_loader)

        # ===== LOG =====
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

    return history

import torch
import torch.nn.functional as F

import copy
from tqdm import tqdm

from src.utils.training import accuracy, get_masks, compute_loss, auc, hits_at_k

from torch_geometric.utils import negative_sampling


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


def train_graph_regression(model, train_loader, val_loader, optimizer, device, mean, std, epochs=200):

    history = {
        "train_loss": [],
        "train_mae": [],
        "val_loss": [],
        "val_mae": []
    }

    for epoch in tqdm(range(epochs)):

        # ===== TRAIN =====
        model.train()

        train_loss = 0
        train_mae = 0

        for data in train_loader:

            data = data.to(device)

            optimizer.zero_grad()

            out = model(data.x, data.edge_index, data.batch)
            # y = data.y.view(-1, 1).float()
            y = (data.y.view(-1, 1).float() - mean) / std

            loss = F.mse_loss(out, y)
            train_loss += loss.item()

            mae = F.l1_loss(out, y, reduction="mean")
            train_mae += mae.item()

        total_train_loss = train_loss / len(train_loader)
        total_train_mae = train_mae / len(train_loader)

        # ===== VALIDATION =====
        model.eval()

        val_loss = 0
        val_mae = 0

        with torch.no_grad():
            for data in val_loader:

                data = data.to(device)

                out = model(data.x, data.edge_index, data.batch)
                # y = data.y.view(-1, 1).float()
                y = (data.y.view(-1, 1).float() - mean) / std

                loss = F.mse_loss(out, y)
                val_loss += loss.item()

                mae = F.l1_loss(out, y, reduction="mean")
                val_mae += mae.item()

        total_val_loss = val_loss / len(val_loader)
        total_val_mae = val_mae / len(val_loader)

        # ===== LOG =====
        history["train_loss"].append(total_train_loss)
        history["train_mae"].append(total_train_mae)
        history["val_loss"].append(total_val_loss)
        history["val_mae"].append(total_val_mae)

    return history


def train_link_prediction(
    model,
    data,
    split_edge,
    optimizer,
    epochs=200
):

    history = {
        "train_loss": [],
        "train_auc": [],
        "train_hits": [],
        "val_loss": [],
        "val_auc": [],
        "val_hits": [],
    }

    for epoch in tqdm(range(epochs)):

        # ===== TRAIN =====
        model.train()
        optimizer.zero_grad()

        z = model(data.x, data.edge_index)

        pos_edge_index = split_edge["train"]["edge"].t()

        neg_edge_index = negative_sampling(
            edge_index=data.edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=pos_edge_index.size(1)
        )

        train_loss, train_pred, train_label = compute_loss(
            z, pos_edge_index, neg_edge_index
        )

        train_loss.backward()
        optimizer.step()

        train_auc = auc(train_pred, train_label)

        train_hits = hits_at_k(
            train_pred[:len(pos_edge_index[0])],
            train_pred[len(pos_edge_index[0]):]
        )

        # ===== VALIDATION =====
        model.eval()
        with torch.no_grad():

            z = model(data.x, data.edge_index)

            pos_edge_index = split_edge["valid"]["edge"].t()
            neg_edge_index = split_edge["valid"]["edge_neg"].t()

            val_loss, val_pred, val_label = compute_loss(
                z, pos_edge_index, neg_edge_index
            )

            val_auc = auc(val_pred, val_label)

            val_hits = hits_at_k(
                val_pred[:len(pos_edge_index[0])],
                val_pred[len(pos_edge_index[0]):]
            )

        # ===== LOG =====
        history["train_loss"].append(train_loss.item())
        history["train_auc"].append(train_auc)
        history["train_hits"].append(train_hits)
        history["val_loss"].append(val_loss.item())
        history["val_auc"].append(val_auc)
        history["val_hits"].append(val_hits)

    return history

import torch.nn.functional as F
from tqdm import tqdm


def train_cora(model, data, optimizer, epochs=200):

    history = {
        "train_loss": [],
        "train_acc": []
    }

    for epoch in tqdm(range(epochs)):

        model.train()

        optimizer.zero_grad()

        out = model(data.x, data.edge_index)

        loss = F.cross_entropy(
            out[data.train_mask],
            data.y[data.train_mask]
        )

        loss.backward()
        optimizer.step()

        pred = out[data.train_mask].argmax(dim=1)
        acc = (pred == data.y[data.train_mask]).float().mean()

        history["train_loss"].append(loss.item())
        history["train_acc"].append(acc.item())

    return history


def train_enzymes(model, loader, optimizer, device, epochs=200):

    history = {
        "train_loss": [],
        "train_acc": []
    }

    for epoch in tqdm(range(epochs)):

        model.train()

        total_loss = 0
        correct = 0
        total = 0

        for data in loader:

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

        acc = correct / total

        history["train_loss"].append(total_loss / len(loader))
        history["train_acc"].append(acc)

    return history

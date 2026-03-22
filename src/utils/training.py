import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


def decode(z, edge_index):
    # dot product
    return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)


def compute_loss(z, pos_edge_index, neg_edge_index):

    pos_pred = decode(z, pos_edge_index)
    neg_pred = decode(z, neg_edge_index)

    pos_label = torch.ones(pos_pred.size(0), device=z.device)
    neg_label = torch.zeros(neg_pred.size(0), device=z.device)

    pred = torch.cat([pos_pred, neg_pred], dim=0)
    label = torch.cat([pos_label, neg_label], dim=0)

    loss = F.binary_cross_entropy_with_logits(pred, label)

    return loss, pred, label


def auc(pred, label):
    pred = pred.sigmoid().detach().cpu().numpy()
    label = label.detach().cpu().numpy()
    return roc_auc_score(label, pred)


def hits_at_k(pos_pred, neg_pred, K=50):
    pos_pred = pos_pred.cpu()
    neg_pred = neg_pred.cpu()

    hits = []

    for i in range(pos_pred.size(0)):
        scores = torch.cat([pos_pred[i].view(1), neg_pred])
        rank = (scores > scores[0]).sum().item() + 1
        hits.append(rank <= K)

    return sum(hits) / len(hits)


def accuracy(pred, labels):
    pred = pred.argmax(dim=1)
    correct = (pred == labels).sum()
    return correct.item() / len(labels)


def get_masks(data, split=0):
    if data.train_mask.dim() == 2:
        train_mask = data.train_mask[:, split]
        val_mask = data.val_mask[:, split]
        test_mask = data.test_mask[:, split]
    else:
        train_mask = data.train_mask
        val_mask = data.val_mask
        test_mask = data.test_mask

    return train_mask, val_mask, test_mask

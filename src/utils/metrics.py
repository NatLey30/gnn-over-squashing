import torch


def accuracy(pred, labels):
    pred = pred.argmax(dim=1)
    correct = (pred == labels).sum()
    return correct.item() / len(labels)

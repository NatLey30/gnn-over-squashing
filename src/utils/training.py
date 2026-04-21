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

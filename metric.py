import numpy as np

from tqdm import tqdm

# heng cher keng's starter kit
def dice_metric_one(pred, target):
    pred = pred.reshape(-1)
    target = target.reshape(-1)

    if target.sum() == 0:
        if pred.sum() == 0:
            return 1
        else:
            return 0

    return (2 * (pred * target).sum()) / (pred + target).sum()


# heng cher keng's starter kit
def dice_metric(preds, targets):
    pos_dices = []
    neg_dices = []
    for i in range(len(preds)):
        dice = dice_metric_one(preds[i], targets[i])

        if targets[i].sum() == 0:
            neg_dices.append(dice)
        else:
            pos_dices.append(dice)

    pos_dices = np.array(pos_dices).mean()
    neg_dices = np.array(neg_dices).mean()

    dice = 0.7886 * neg_dices + (1 - 0.7886) * pos_dices

    return dice


def threshold(preds, targets, thresholds):
    dices = []
    for threshold in thresholds:
        preds_thresholded = preds > threshold
        dices.append(dice_metric(preds_thresholded, targets))

    return np.array(dices)

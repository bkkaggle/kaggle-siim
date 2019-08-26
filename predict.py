import os
import fire
import subprocess

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
import mlcrate as mlc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from pytorch_zoo.utils import seed_environment, notify, load_model

from config import *
from dataset import SIIMDataset
from model import DeepLabV3, ResUNet34
from utils import device, mask_to_rle

seed_environment(SEED)

# test predictions must be in the original resolution


def predict(
    threshold,
    subset=SUBSET,
    key=KEY,
    fold=FOLD,
    height=1024,
    width=1024,
    imgs_folder=TEST_IMGS_FOLDER,
):
    dataset = SIIMDataset(
        height, width, imgs_folder, mode="test", subset=subset, fold=fold
    )

    # model = DeepLabV3(num_classes=1).to(device)
    model = ResUNet34(3, 1).to(device)
    model = load_model(model, fold).eval()

    rles = []

    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataset), total=(len(dataset))):
            img, img_id = batch[0].to(device).unsqueeze(0), batch[1]

            out = model(img)
            # out = torch.sigmoid(out["out"])
            out = torch.sigmoid(out)
            out = out.cpu().numpy().reshape(height, width)
            # must transpose predictions since masks were transposed when preprocessed
            mask = (out > threshold).astype(np.uint8).T

            if np.sum(mask) == 0:
                rle = " -1"
            else:
                rle = mask_to_rle(mask, height, width)

            rles.append([img_id, rle])

    sub = pd.DataFrame(rles, columns=["ImageId", "EncodedPixels"])
    sub.to_csv("submission.csv", index=False)

    notify({"value1": "Submission ready", "value2": ""}, KEY)


if __name__ == "__main__":
    fire.Fire(predict)

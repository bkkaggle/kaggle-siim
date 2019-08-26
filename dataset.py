import os
import glob
import numpy as np
import pandas as pd

import cv2

from sklearn.model_selection import KFold

import torch
from torch.utils.data import Dataset

from config import *


class SIIMDataset(Dataset):
    def __init__(
        self,
        height,
        width,
        imgs_folder,
        masks_folder=None,
        mode="train",
        seed=SEED,
        n_splits=N_SPLITS,
        subset=SUBSET,
        fold=FOLD,
    ):
        self.height = height
        self.width = width

        self.imgs_folder = imgs_folder
        self.masks_folder = masks_folder

        self.mode = mode
        self.n_splits = n_splits
        self.subset = subset
        self.seed = seed
        self.fold = fold

        if self.height == 256 and self.width == 256:
            self.means = [0.384, 0.488, 0.603]
            self.stddevs = [0.192, 0.173, 0.232]
        elif self.height == 512 and self.width == 512:
            self.means = [0.383, 0.487, 0.603]
            self.stddevs = [0.192, 0.173, 0.233]
        elif self.height == 1024 and self.width == 1024:
            self.means = [0.383, 0.486, 0.602]
            self.stddevs = [0.193, 0.174, 0.233]

        rles = pd.read_csv(
            f"./{'train-rle' if mode == 'train' or mode == 'val' else 'submission'}.csv"
        )
        rles = rles.drop_duplicates("ImageId")

        img_ids = list(rles["ImageId"].values)

        if self.subset:
            img_ids = img_ids[:128]

        if self.mode == "train" or self.mode == "val":
            kfold = KFold(n_splits=self.n_splits, random_state=self.seed)
            train_idxs, val_idxs = list(kfold.split(img_ids))[self.fold]

            if mode == "train":
                self.img_ids = np.array(img_ids)[train_idxs]
            else:
                self.img_ids = np.array(img_ids)[val_idxs]
        else:
            self.img_ids = np.array(img_ids)

        print(f"Loaded {mode} dataset with {len(self.img_ids)} images")

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]

        img_path = f"{self.imgs_folder}/{img_id}.png"

        img = cv2.imread(img_path)
        # cv2 expects shapes in (width, height)
        img = cv2.resize(img, (self.width, self.height))
        img = img.reshape(3, self.height, self.width)
        img = img.astype(np.float32)
        img /= 255

        img[0] -= self.means[0]
        img[1] -= self.means[1]
        img[2] -= self.means[2]

        img[0] /= self.stddevs[0]
        img[1] /= self.stddevs[1]
        img[2] /= self.stddevs[2]

        img = torch.from_numpy(img)

        if self.mode == "test":
            return img, img_id
        else:
            masks = glob.glob(f"{self.masks_folder}/{img_id}*")
            img_mask = np.zeros((1, self.height, self.width), dtype=np.float32)
            for i in range(len(masks)):
                mask_path = f"{self.masks_folder}/{img_id}-{i}.png"
                mask = cv2.imread(mask_path)
                mask = mask[:, :, 0]
                # cv2 expects shapes in (width, height)
                mask = cv2.resize(mask, (self.width, self.height))
                mask = mask.reshape(1, self.height, self.width)
                mask = mask.astype(np.float32)
                mask /= 255

                img_mask += mask

            img_mask = np.ceil(img_mask)
            mask = torch.from_numpy((img_mask > 0).astype(float)).float()

            return img, mask

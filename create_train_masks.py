import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2


def rle2mask(rle, width, height):
    mask = np.zeros(width * height)
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]
    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position:current_position+lengths[index]] = 1
        current_position += lengths[index]
    return mask.reshape(width, height).T


rles = pd.read_csv('./train-rle.csv')

for uid in tqdm(np.unique(rles['ImageId'].values)):
    rle_masks = rles.loc[rles['ImageId'] == uid].values[:, 1]
    
    masks = []

    for i in range(len(rle_masks)):
        rle_mask = rle_masks[i]
        if rle_mask == ' -1':
            rle_mask = np.zeros((1024, 1024), dtype=np.uint8)
        else:
            rle_mask = rle2mask(rle_mask, 1024, 1024).astype(np.uint8) * 255
        
        cv2.imwrite(f'./train_masks/{uid}-{i}.png', rle_mask)
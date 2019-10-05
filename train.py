
import os
import fire
import subprocess

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
import mlcrate as mlc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from apex import amp

# save_model, gpu_usage
from pytorch_zoo.utils import seed_environment, notify, load_model, n_params

from config import *
from dataset import SIIMDataset
from loss import DiceLoss
from model import DeepLabV3, ResUNet34
from metric import dice_metric, threshold
from utils import device, save_model

seed_environment(SEED)


def train(checkpoint=None, log_more=LOG_MORE, seed=SEED, n_splits=N_SPLITS, subset=SUBSET, key=KEY, fold=FOLD, height=HEIGHT, width=WIDTH, imgs_folder=IMGS_FOLDER, masks_folder=MASKS_FOLDER, batch_size=BATCH_SIZE, workers=WORKERS, epochs=EPOCHS, gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS, early_stopping=EARLY_STOPPING, histogram_interval=HISTOGRAM_INTERVAL):
    timer = mlc.time.Timer()
    writer = SummaryWriter()

    writer.add_text('Checkpoint', str(checkpoint), 0)
    writer.add_text('Mode', 'Train', 0)
    writer.add_text('Log more', str(log_more), 0)
    writer.add_text('Seed', str(seed), 0)
    writer.add_text('n splits', str(n_splits), 0)
    writer.add_text('Subset', str(subset), 0)
    writer.add_text('Key', str(key), 0)
    writer.add_text('Fold', str(fold), 0)
    writer.add_text('Height', str(height), 0)
    writer.add_text('Width', str(width), 0)
    writer.add_text('Imgs folder', str(imgs_folder), 0)
    writer.add_text('Masks folder', str(masks_folder), 0)
    writer.add_text('Batch size', str(batch_size), 0)
    writer.add_text('Workers', str(workers), 0)
    writer.add_text('Epochs', str(epochs), 0)
    writer.add_text('Gradient accumulation steps',
                    str(gradient_accumulation_steps), 0)
    writer.add_text('Early stopping', str(early_stopping), 0)
    writer.add_text('Histogram interval', str(histogram_interval), 0)

    train_dataset = SIIMDataset(height, width, imgs_folder, masks_folder,
                                mode='train', seed=seed, n_splits=n_splits, subset=subset, fold=fold)
    val_dataset = SIIMDataset(height, width, imgs_folder, masks_folder,
                              mode='val', seed=seed, n_splits=n_splits, subset=subset, fold=fold)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)

    # model = DeepLabV3(num_classes=1).to(device)
    model = ResUNet34(3, 1).to(device)

    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint))
        print(
            f'Loaded model parameters from pretrained checkpoint {checkpoint}')

    print(f'Model parameters: {n_params(model)}')

    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
    criterion = nn.BCEWithLogitsLoss()  # DiceLoss()

    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    thresholds = np.arange(0.01, 1, 0.1)

    best_train_loss = 1e6
    best_val_loss = 1e6

    best_train_metric = 1e-6
    best_val_metric = 1e-6

    best_val_threshold = 0
    best_val_threshold_scores = np.zeros(len(thresholds))

    best_epoch = 0

    global_train_step = 0
    global_val_step = 0

    for epoch in range(epochs):
        print(f'\nStarting epoch {epoch}')
        timer.add(epoch)

        train_loss = 0
        val_loss = 0

        train_metric = 0
        val_metric = 0
        val_threshold = 0

        val_threshold_scores = np.zeros(len(thresholds))

        model.train()
        for i, batch in tqdm(enumerate(train_dataloader), total=(int(len(train_dataset) / batch_size))):
            img, mask = batch[0].to(device), batch[1].to(device)

            if img.shape[0] == 1:
                continue

            if log_more and i < 3:
                print('\n')
                print(subprocess.run(['nvidia-smi'],
                                     stdout=subprocess.PIPE).stdout)
                print('\n')

            out = model(img)
            # only necessary for deeplab models
            # out = out['out']

            loss = criterion(out, mask)
            loss = loss / gradient_accumulation_steps
            
            train_loss += loss.item()

            writer.add_scalar('train_loss', loss.item(), global_train_step)

            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()

            # must be called between calls to optimizer.step() and optimizer.zero_grad()
            if (i + 1) % (histogram_interval * gradient_accumulation_steps) == 0:
                for name, param in model.named_parameters():
                    writer.add_histogram(
                        f'{name}', param, global_train_step, bins='sqrt')
                    if param.grad is not None:
                        writer.add_histogram(
                            f'{name}.grad', param.grad, global_train_step, bins='sqrt')

            if (i + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            global_train_step += 1

        val_preds = []
        y_val = []

        model.eval()
        with torch.no_grad():
            for j, batch in tqdm(enumerate(val_dataloader), total=(int(len(val_dataset) / batch_size))):
                img, mask = batch[0].to(device), batch[1].to(device)

                out = model(img)
                # only necessary for deeplab models
                # out = out['out']

                loss = criterion(out, mask)
                val_loss += loss.item()

                out = torch.sigmoid(out)

                val_preds.append(out.detach().cpu().numpy())
                y_val.append(mask.detach().cpu().numpy())

                writer.add_scalar('val_loss', loss.item(), global_val_step)

                global_val_step += 1

        train_loss /= (i + 1)
        val_loss /= (j + 1)

        val_preds = np.concatenate(val_preds, axis=0)
        y_val = np.concatenate(y_val, axis=0)

        val_threshold_scores = threshold(val_preds, y_val, thresholds)
        val_metric = np.max(val_threshold_scores)
        val_threshold = thresholds[np.argmax(val_threshold_scores)]

        writer.add_scalar('train_epoch_loss', train_loss, epoch)
        writer.add_scalar('val_epoch_loss', val_loss, epoch)

        writer.add_scalar('val_epoch_metric', val_metric, epoch)
        writer.add_scalar('val_epoch_threshold', val_threshold, epoch)

        message = f'Finished epoch {epoch} in {timer.fsince(epoch)} | Train loss: {train_loss} | Val loss: {val_loss} | Train metric: {train_metric} | Val metric: {val_metric} | Val threshold: {val_threshold}'
        print(message)

        if log_more:
            notify({'value1': f'Epoch {epoch} finished', 'value2': message}, key)

        # Choose one
        # if val_metric > best_val_metric:
        if val_loss < best_val_loss:
            best_epoch = epoch

            best_train_loss = train_loss
            best_val_loss = val_loss

            best_train_metric = train_metric
            best_val_metric = val_metric

            best_val_threshold = val_threshold
            best_val_threshold_scores = val_threshold_scores

            save_model(model, fold=fold)

            if log_more:
                notify({'value1': f'Saving model',
                        'value2': f'Saving model at epoch {epoch}'}, key)

        elif epoch - best_epoch > early_stopping:
            message = f'The val loss has not decreased for {early_stopping} epochs, stopping training'
            print(message)

            if log_more:
                notify({'value1': 'Stopping training', 'value2': message}, key)

            break

    for i in range(len(thresholds)):
        writer.add_scalar('averaged val threshold scores over batches of the best epoch x thresholds',best_val_threshold_scores[i], int(thresholds[i] * 100))

    message = f'\n\nFinished training {epoch + 1} epochs in {timer.fsince(0)} | Best epoch: {best_epoch} | best train loss: {best_train_loss} | best val loss: {best_val_loss} | best train metric: {best_train_metric} | best val metric: {best_val_metric} | best val threshold: {best_val_threshold}'
    print(message)
    notify({'value1': 'Finished training', 'value2': message}, key)


if __name__ == '__main__':
    fire.Fire(train)

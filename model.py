# coding: utf-8
import time
import math
import os

import itertools
from functools import partial
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
# from torchvision import transforms as T

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from data_utils import get_time_series
from mem_transformer import MemTransformerLM

from utils.exp_utils import create_exp_dir
from utils.data_parallel import BalancedDataParallel
from utils.initialization import weights_init

import argparse
from utils.argparsing import parser
from utils.torch_utils import non_emb_param_count, openai_compute


###############################################################################
##
## Helper functions
##
def update_dropout(m):
    classname = m.__class__.__name__
    if classname.find("Dropout") != -1:
        if hasattr(m, "p"):
            m.p = args.dropout
    if hasattr(m, "dropout_p"):
        m.dropout_p = args.dropatt


def update_dropatt(m):
    if hasattr(m, "dropatt"):
        m.dropatt.p = args.dropatt
    if hasattr(m, "dropatt_p"):
        m.dropatt_p = args.dropatt


################################################################################
##
## Dataset class
##
class TimeSeries(Dataset):
    def __init__(self, data: pd.DataFrame, mode='train'):
        global globalState

        self.data = data

        # In debug mode, only use about 2 epoch of data
        # TODO refactor to use exactly 2 epoch
        self.debug = globalState.debug
        if globalState.debug:
            self.data = data[:700]

        self.len = self.data.shape[0]
        self.size = 256

    def _get_img_path(self, index, channel):
        experiment, well, plate = self.records[index].experiment, self.records[
            index].well, self.records[index].plate
        return '/'.join([self.img_dir,
                         self.mode,
                         experiment,
                         'Plate{}'.format(plate),
                         '{}_s{}_w{}.png'.format(well,
                                                 self.site,
                                                 channel)])

    def __getitem__(self, index):
        paths = [self._get_img_path(index, ch) for ch in self.channels]
        img = torch.cat([self._load_img_as_tensor(img_path)
                         for img_path in paths])
        if self.mode == 'train':
            return img, self.records[index].sirna
        else:
            return img, self.records[index].id_code

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len


################################################################################
##
## Lightning module of the model
##
class TransformerXL(pl.LightningModule):

    def __init__(self, dataFrame: pd.DataFrame,
                 model: MemTransformerLM):
        super(TransformerXL, self).__init__()
        global globalState

        # not the best model...
        self.data = dataFrame
        self.train_loader = DataLoader(dataFrame[:-1],
                                       batch_size=globalState.d_batch,
                                       num_workers=4 - 1)
        self.val_loader = DataLoader(dataFrame, batch_size=globalState.d_batch,
                                     pin_memory=True,
                                     num_workers=4 - 1)
        self.test_loader = DataLoader(dataFrame,
                                      batch_size=globalState.d_batch,
                                      num_workers=4 - 1)

        self.model = model
        self.criteria = nn.CrossEntropyLoss()

    # prepare_data() makes sure that data is available for training
    # We assume that data was downloaded, save as a pickle, NaN not changed yet.
    # Create clean data set with NaN -> 0
    # WARNING Change when using market indices as well as returns (cannot fill
    # NaN indices with 0)
    def prepare_data(self) -> None:
        global globalState

        data_set = pd.read_pickle(f"{globalState.datadir}/allData.pickle")
        data_set.fillna(0, inplace=True)
        data_set.to_pickle(f"{globalState.datadir}/allDataClean.pickle")
        return None

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        # REQUIRED
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criteria(y_hat, y)
        loss = loss.unsqueeze(dim=-1)
        return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        x, y = batch
        y_hat = self.forward(x)
        val_loss = self.criteria(y_hat, y)
        val_loss = val_loss.unsqueeze(dim=-1)
        return {'val_loss': val_loss}

    def validation_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'avg_val_loss': avg_loss}

    def configure_optimizers(self):
        # REQUIRED
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20,
                                                    gamma=0.1)
        return [optimizer], [scheduler]

    @pl.data_loader
    def tng_dataloader(self):
        # REQUIRED
        return self.train_loader

    @pl.data_loader
    def val_dataloader(self):
        # OPTIONAL
        return self.val_loader

    @pl.data_loader
    def test_dataloader(self):
        # OPTIONAL
        pass


################################################################################
##
## Checkpoint callback to save best model like keras.
##
checkpoint_callback = ModelCheckpoint(
    filepath='../working',
    save_best_only=True,
    verbose=True,
    monitor='avg_val_loss',
    mode='min'
)

################################################################################
##
## Main.
##
globalState = None

if __name__ == "__main__":
    args = parser.parse_args()

    global globalState
    globalState = args

    # TODO train_ts(globalState)

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
from torch.utils.data import IterableDataset
from torch.utils.data import Dataset

# from torchvision import transforms as T

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from mem_transformer import MemTransformerLM

from utils.exp_utils import create_exp_dir
from utils.data_parallel import BalancedDataParallel
from utils.initialization import weights_init

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
class IterableTimeSeries(Dataset):
    def __init__(self, data: pd.DataFrame, mode='train'):
        global globalState

        # Keeps the size of a batch to start the test set
        self.n_batch = globalState['n_batch']

        # In debug mode, only use about 2 epoch of data
        # TODO refactor to use exactly 2 epoch instead of 700 dates.
        self.debug = globalState['debug']
        if globalState['debug']:
            actual_data = data[:700]
        else:
            actual_data = data

        # Adjust the start of the dataset for training / val / test
        if mode == 'train':
            start_index = 0
        elif mode == 'val':
            start_index = globalState['n_model']
        elif mode == 'val':
            start_index = globalState['n_model'] + globalState['n_val']

        # This is the actual data on which to iterate
        self.data = actual_data[start_index:, :]

        # d_series is the depth of a series (how many data points per dates)
        # n_series is the number of series (how many dates)
        self.n_series, self.d_series = data.shape

        # Each training data point is a set of series to fill the model:
        # One date (d_series data points) for each entry to the model, that is
        # globalState.n_model
        self.n_model = globalState['n_model']

    def __getitem__(self, index):
        return self.data[index:index + self.n_model - 1, :], \
               self.data[index + self.n_model, :]

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.n_series


################################################################################
##
## Lightning module of the model
##
class TransformerXL(pl.LightningModule):

    def __init__(self, dataFrame: pd.DataFrame):
        super(TransformerXL, self).__init__()
        global globalState

        self.data = dataFrame

        self.model = MemTransformerLM(n_token=globalState['n_token'],
                                      n_layer=globalState['n_layer'],
                                      n_head=globalState['n_head'],
                                      n_model=globalState['n_model'],
                                      d_head=globalState['d_head'],
                                      d_inner=globalState['d_inner'],
                                      dropout=globalState['dropout'],
                                      dropatt=globalState['dropatt'],
                                      tie_weight=globalState['tie_weight'],
                                      d_embed=globalState['d_embed'],
                                      div_val=globalState['div_val'],
                                      tie_projs=globalState['tie_projs'],
                                      pre_lnorm=globalState['pre_lnorm'],
                                      tgt_len=globalState['tgt_len'],
                                      ext_len=globalState['ext_len'],
                                      mem_len=globalState['mem_len'],
                                      cutoffs=globalState['cutoffs'],
                                      adapt_inp=globalState['adapt_inp'],
                                      same_length=globalState['same_length'],
                                      clamp_len=globalState['clam_len']
                                      )
        self.criteria = nn.CrossEntropyLoss()

    ############################################################################
    ##
    ## STEP 1: Define the model
    ##
    ############################################################################

    def forward(self, x):
        return self.model(x)

    ############################################################################
    ##
    ## STEP 2: Prepare a dataset that will be available to the dataloaders
    ##
    ############################################################################

    # prepare_data() makes sure that data is available for training
    # We assume that data was downloaded, save as a pickle, NaN not changed yet.
    # Create clean data set with NaN -> 0
    # WARNING Change when using market indices as well as returns (cannot fill
    # NaN indices with 0)
    def prepare_data(self) -> None:
        global globalState

        data_set = pd.read_pickle(f"{globalState['datadir']}/allData.pickle")
        data_set.fillna(0, inplace=True)
        data_set.to_pickle(f"{globalState['datadir']}/allDataClean.pickle")
        return None

    ############################################################################
    ##
    ## STEP 3: Configure the optimizer (how to use the data in the model)
    ##
    ############################################################################

    def configure_optimizers(self):
        # REQUIRED
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20,
                                                    gamma=0.1)
        return [optimizer], [scheduler]

    ############################################################################
    ##
    ## STEP 4: How to train the model: first a dalaloader, the how to train
    ##
    ############################################################################

    def train_loader(self):
        global globalState

        return DataLoader(IterableTimeSeries(self.data, mode='train'),
                          batch_size=globalState['n_batch'], num_workers=4 - 1)

    def training_step(self, batch, batch_nb):
        # REQUIRED
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criteria(y_hat, y)
        loss = loss.unsqueeze(dim=-1)
        return {'loss': loss}

    ############################################################################
    ##
    ## STEP 5: Then validate on a validation set dataloader
    ##
    ############################################################################

    def val_loader(self):
        global globalState

        return DataLoader(IterableTimeSeries(self.data, mode='val'),
                          batch_size=globalState['n_batch'], num_workers=4 - 1)

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

    # def validation_epoch_end(self, outputs):
    #     avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
    #     tensorboard_logs = {'val_loss': avg_loss}
    #     return {'val_loss': avg_loss, 'log': tensorboard_logs}

    ############################################################################
    ##
    ## STEP 6: Finally test the model
    ##
    ############################################################################

    def train_loader(self):
        global globalState

        return DataLoader(IterableTimeSeries(self.data, mode='test'),
                          batch_size=globalState['n_batch'], num_workers=4 - 1)


    # def test_step(self, batch, batch_idx):
    #     x, y = batch
    #     logits = self(x)
    #     loss = F.nll_loss(logits, y)
    #     return {'val_loss': loss}
    #
    # def test_epoch_end(self, outputs):
    #     avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
    #     tensorboard_logs = {'val_loss': avg_loss}
    #     return {'val_loss': avg_loss, 'log': tensorboard_logs}


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

# TODO Replace by default dictionary
globalState = None

if __name__ == "__main__":
    args = parser.parse_args()

    global globalState
    globalState = args.__dict__

    # TODO train_ts(globalState)

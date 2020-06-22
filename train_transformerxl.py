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

from TransformerXL_model import TransformerXL

from utils.exp_utils import create_exp_dir, logging
from utils.data_parallel import BalancedDataParallel
from utils.initialization import weights_init

from utils.argparsing import parser
from utils.torch_utils import non_emb_param_count, openai_compute


###############################################################################
##
## Helper functions
##
def update_dropout(globalState, m):
    classname = m.__class__.__name__
    if classname.find("Dropout") != -1:
        if hasattr(m, "p"):
            m.p = globalState["dropout"]
    if hasattr(m, "dropout_p"):
        m.dropout_p = globalState["dropatt"]


def update_dropatt(globalState, m):
    if hasattr(m, "dropatt"):
        m.dropatt.p = globalState["dropatt"]
    if hasattr(m, "dropatt_p"):
        m.dropatt_p = globalState["dropatt"]


################################################################################
##
## Dataset class
##
class IterableTimeSeries(Dataset):
    def __init__(self, globalState, data: pd.DataFrame, mode="train"):

        # Keeps the size of a batch to start the test set
        self.n_batch = globalState["n_batch"]

        # In debug mode, only use about 2 epoch of data
        # TODO refactor to use exactly 2 epoch instead of 700 dates.
        self.debug = globalState["debug"]
        if globalState["debug"]:
            actual_data = data[:700]
        else:
            actual_data = data

        # Adjust the start of the dataset for training / val / test
        if mode == "train":
            start_index = 0
        elif mode == "val":
            start_index = globalState["n_model"]
        elif mode == "test":
            start_index = globalState["n_model"] + globalState["n_val"]

        # This is the actual data on which to iterate
        self.data = actual_data[start_index:, :]

        # d_series is the depth of a series (how many data points per dates)
        # n_series is the number of series (how many dates)
        self.n_series, self.d_series = data.shape

        # Each training data point is a set of series to fill the model:
        # One date (d_series data points) for each entry to the model, that is
        # globalState.n_model
        self.n_model = globalState["n_model"]

    def __getitem__(self, index):
        return (
            self.data[index : index + self.n_model - 1, :],
            self.data[index + self.n_model, :],
        )

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.n_series


################################################################################
##
## Lightning module of the model
##
class Train_TransformerXL(pl.LightningModule):
    def __init__(self, globalState):
        super(Train_TransformerXL, self).__init__()

        self.globalState = globalState

        self.model = TransformerXL(
            d_model=globalState["d_model"],
            n_layer=globalState["n_layer"],
            n_head=globalState["n_head"],
            n_model=globalState["n_model"],
            d_head=globalState["d_head"],
            d_inner=globalState["d_inner"],
            dropout=globalState["dropout"],
            dropatt=globalState["dropatt"],
            d_embed=globalState["d_embed"],
            pre_lnorm=globalState["pre_lnorm"],
            tgt_len=globalState["tgt_len"],
            ext_len=globalState["ext_len"],
            mem_len=globalState["mem_len"],
            cutoffs=globalState["cutoffs"],
            adapt_inp=globalState["adapt_inp"],
            same_length=globalState["same_length"],
            clamp_len=globalState["clamp_len"],
        )
        self.criteria = nn.CrossEntropyLoss()

    ############################################################################
    #
    # STEP 1: Define the model
    #
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
        data_set = pd.read_pickle(f"{self.globalState['datadir']}/allData.pickle")
        data_set.fillna(0, inplace=True)
        data_set.to_pickle(f"{self.globalState['datadir']}/allDataClean.pickle")
        return None

    ############################################################################
    #
    # STEP 3: Configure the optimizer (how to use the data in the model)
    #
    ############################################################################

    ############################################################################
    #
    # STEP 3.1: Build a scheduler

    def build_scheduler(self, optimizers):
        optimizer, optimizer_sparse = optimizers
        scheduler_sparse = None

        if self.globalState["scheduler"] == "cosine":
            # here we do not set eta_min to lr_min to be backward compatible
            # because in previous versions eta_min is default to 0
            # rather than the default value of lr_min 1e-6
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                self.globalState["max_step"],
                eta_min=self.globalState["eta_min"],
            )  # should use eta_min arg

        elif self.globalState["scheduler"] == "inv_sqrt":
            # originally used for Transformer (in Attention is all you need)
            def lr_lambda(step):
                # return a multiplier instead of a learning rate
                if step == 0 and self.globalState["warmup_step"] == 0:
                    return 1.0
                else:
                    return (
                        1.0 / (step ** 0.5)
                        if step > self.globalState["warmup_step"]
                        else step / (self.globalState["warmup_step"] ** 1.5)
                    )

            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        elif self.globalState["scheduler"] == "dev_perf":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=self.globalState["decay_rate"],
                patience=self.globalState["patience"],
                min_lr=self.globalState["lr_min"],
            )

        elif self.globalState["scheduler"] == "constant":
            pass

        else:
            raise ValueError(
                f"scheduler type {self.globalState['scheduler']} not recognized"
            )

        return scheduler, scheduler_sparse

    ############################################################################
    #
    # STEP 3.2: Build an optimizer

    def build_optimizer(self, reload=False):
        optimizer_sparse = None
        if self.globalState["optim"].lower() == "sgd":
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.globalState["lr"],
                momentum=self.globalState["mom"],
            )
        elif self.globalState["optim"].lower() == "adam":
            optimizer = optim.Adam(self.model.parameters(), lr=self.globalState["lr"])
        elif self.globalState["optim"].lower() == "adagrad":
            optimizer = optim.Adagrad(
                self.model.parameters(), lr=self.globalState["lr"]
            )
        else:
            raise ValueError(
                f"optimizer type {self.globalState['optim']} not recognized"
            )

        if reload:
            if self.globalState["restart_from"] is not None:
                optim_name = f"optimizer_{self.globalState['restart_from']}.pt"
            else:
                optim_name = "optimizer.pt"
            optim_file_name = os.path.join(self.globalState["restart_dir"], optim_name)
            logging(f"reloading {optim_file_name}")
            if os.path.exists(
                os.path.join(self.globalState["restart_dir"], optim_name)
            ):
                with open(
                    os.path.join(self.globalState["restart_dir"], optim_name), "rb"
                ) as optim_file:
                    opt_state_dict = torch.load(optim_file)
                    try:
                        optimizer.load_state_dict(opt_state_dict)
                    # in case the optimizer param groups aren't the same shape,
                    # merge them
                    except:
                        logging("merging optimizer param groups")
                        opt_state_dict["param_groups"][0]["params"] = [
                            param
                            for param_group in opt_state_dict["param_groups"]
                            for param in param_group["params"]
                        ]
                        opt_state_dict["param_groups"] = [
                            opt_state_dict["param_groups"][0]
                        ]
                        optimizer.load_state_dict(opt_state_dict)
            else:
                logging("Optimizer was not saved. Start from scratch.")

        return optimizer, optimizer_sparse

    def configure_optimizers(self):
        optimizer = self.build_optimizer()
        scheduler = self.build_scheduler(optimizer)
        return [optimizer], [scheduler]

    ############################################################################
    #
    # STEP 4: How to train the model: first a dalaloader, the how to train
    #
    ############################################################################

    def train_loader(self):
        return DataLoader(
            IterableTimeSeries(self.globalState, self.data, mode="train"),
            batch_size=globalState["n_batch"],
            num_workers=4 - 1,
        )

    def training_step(self, batch, batch_nb):
        # REQUIRED
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criteria(y_hat, y)
        loss = loss.unsqueeze(dim=-1)
        return {"loss": loss}

    ############################################################################
    #
    # STEP 5: Then validate on a validation set dataloader
    #
    ############################################################################

    def val_loader(self):
        return DataLoader(
            IterableTimeSeries(self.globalState, self.data, mode="val"),
            batch_size=globalState["n_batch"],
            num_workers=4 - 1,
        )

    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        x, y = batch
        y_hat = self.forward(x)
        val_loss = self.criteria(y_hat, y)
        val_loss = val_loss.unsqueeze(dim=-1)
        return {"val_loss": val_loss}

    def validation_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        return {"avg_val_loss": avg_loss}

    # def validation_epoch_end(self, outputs):
    #     avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
    #     tensorboard_logs = {'val_loss': avg_loss}
    #     return {'val_loss': avg_loss, 'log': tensorboard_logs}

    ############################################################################
    ##
    ## STEP 6: Finally test the model
    ##
    ############################################################################

    def test_loader(self):
        return DataLoader(
            IterableTimeSeries(self.globalState, self.data, mode="test"),
            batch_size=globalState["n_batch"],
            num_workers=4 - 1,
        )

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
    filepath="../working",
    save_top_k=1,
    verbose=True,
    monitor="avg_val_loss",
    mode="min",
)

################################################################################
##
## Main.
##
if __name__ == "__main__":
    args = parser.parse_args()

    globalState = args.__dict__

    train_transformerxl = Train_TransformerXL(globalState)


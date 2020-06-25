# coding: utf-8

import os, inspect

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from TXL_modules import Transformer_XL

from utils.exp_utils import logging
from utils.utils import GlobalState


###############################################################################
##
## Helper functions
##
def update_dropout(global_state: GlobalState, m):
    classname = m.__class__.__name__
    if classname.find("Dropout") != -1:
        if hasattr(m, "p"):
            m.p = global_state.dropout
    if hasattr(m, "dropout_p"):
        m.dropout_p = global_state.dropout_attn


def update_dropatt(global_state: GlobalState, m):
    if hasattr(m, "dropatt"):
        m.dropout_attn.p = global_state.dropout_attn
    if hasattr(m, "dropatt_p"):
        m.dropatt_p = global_state.dropout_attn


################################################################################
##
## Dataset class
##
class IterableTimeSeries(Dataset):
    def __init__(self, global_state: GlobalState, data, mode="train"):
        super(IterableTimeSeries, self).__init__()

        # Keeps the size of a batch to start the test set
        self.n_batch = global_state.n_batch

        # In debug mode, only use about 2 epoch of input
        # TODO refactor to use exactly 2 epoch instead of 700 dates.
        self.debug = global_state.debug
        if global_state.debug:
            actual_data = data[0:700, :]
        else:
            actual_data = data

        # Adjust the start of the dataset for training / val / test
        if mode == "train":
            start_index = 0
        elif mode == "val":
            start_index = global_state.n_model
        elif mode == "test":
            start_index = global_state.n_model + global_state.n_val

        # This is the actual input on which to iterate
        self.data = actual_data[start_index:, :]

        # d_series is the depth of a series (how many input points per dates)
        # n_series is the number of series (how many dates)
        self.n_series, self.d_series = data.shape

        # Each training input point is a set of series to fill the transformer_model:
        # One date (d_series input points) for each entry to the transformer_model, that is
        # global_state.d_model
        self.n_model = global_state.n_model

    def __getitem__(self, index):
        # An item is a tuple of:
        #   - a transformer_model input being, say, 60 dates of time series
        #   -  the following date as expected output
        return (self.data[index: index + self.n_model, :],
                self.data[index + self.n_model, :])

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.n_series


################################################################################
##
## Lightning module of the transformer_model
##
class TransformerXL_Trainer(pl.LightningModule):
    def __init__(self, global_state: GlobalState):
        super(TransformerXL_Trainer, self).__init__()

        self.global_state = global_state

        self.transformer_model = Transformer_XL(n_layer=global_state.n_layer,
                                                d_hidden=global_state.d_hidden,
                                                d_pos_enc=global_state.d_pos_enc,
                                                n_head=global_state.n_head,
                                                d_head=global_state.d_head,
                                                d_FF_inner=global_state.d_FF_inner,
                                                d_model=global_state.d_model,
                                                dropout=global_state.dropout,
                                                dropout_attn=global_state.dropout_attn,
                                                n_model=global_state.n_model,
                                                n_mems=global_state.n_mems,
                                                debug=global_state.debug)
        self.criteria = nn.CrossEntropyLoss()

    ############################################################################
    #
    # STEP 1: Define the transformer_model
    #
    ############################################################################

    def forward(self, input: torch.FloatTensor, output: torch.FloatTensor,
                *mems):
        if self.global_state.debug:
            logging(f"")
            logging(f"")
            logging(f"########################################################")
            logging(f"")
            logging(f"{self.__class__.__name__}, "
                    f"{inspect.currentframe().f_code.co_name}: "
                    f" input: {input.size()}")
            logging(f"{self.__class__.__name__}, "
                    f"{inspect.currentframe().f_code.co_name}: "
                    f" output: {output.size()}")
            logging(f"{self.__class__.__name__}, "
                    f"{inspect.currentframe().f_code.co_name}: "
                    f" mems: {len(mems)}")

        return self.transformer_model(input, output, *mems)

    ############################################################################
    ##
    ## STEP 2: Prepare a dataset that will be available to the dataloaders
    ##
    ############################################################################

    # prepare_data() makes sure that input is available for training
    # We assume that input was downloaded, save as a pickle, NaN not changed yet.
    # The input loaders will: create clean input set with NaN -> 0 and remove the input index
    # WARNING Change when using market indices as well as returns (cannot fill
    # NaN indices with 0)
    def prepare_data(self) -> None:
        data_set = pd.read_pickle(
            f"{self.global_state.data_dir}/allData.pickle")

        # Does nothing.
        data_set.to_pickle(f"{self.global_state.data_dir}/allDataClean.pickle")
        return None

    ############################################################################
    #
    # STEP 3: Configure the optimizer (how to use the input in the transformer_model)
    #
    ############################################################################

    #
    # STEP 3.1: Build an optimizer
    #
    def build_optimizer(self, reload=False):
        if self.global_state.optim.lower() == "sgd":
            optimizer = optim.SGD(
                self.transformer_model.parameters(),
                lr=self.global_state.lr,
                momentum=self.global_state.mom,
            )
        elif self.global_state.optim.lower() == "adam":
            optimizer = optim.Adam(self.transformer_model.parameters(),
                                   lr=self.global_state.lr)

        elif self.global_state.optim.lower() == "adagrad":
            optimizer = optim.Adagrad(self.transformer_model.parameters(),
                                      lr=self.global_state.lr)
        else:
            raise ValueError(
                f"optimizer type {self.global_state.optim} not recognized")

        if reload:
            if self.global_state.restart_from is not None:
                optim_name = f"optimizer_{self.global_state.restart_from}.pt"
            else:
                optim_name = "optimizer.pt"

            optim_file_name = os.path.join(self.global_state.restart_dir,
                                           optim_name)
            logging(f"reloading {optim_file_name}")
            if os.path.exists(
                    os.path.join(self.global_state.restart_dir, optim_name)):
                with open(
                        os.path.join(self.global_state.restart_dir, optim_name),
                        "rb"
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

        return optimizer

    #
    # STEP 3.2: Build a scheduler
    #
    def build_scheduler(self, optimizer):
        if self.global_state.scheduler == "cosine":
            # here we do not set eta_min to lr_min to be backward compatible
            # because in previous versions eta_min is default to 0
            # rather than the default value of lr_min 1e-6
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                self.global_state.max_step,
                eta_min=self.global_state.eta_min,
            )

        elif self.global_state.scheduler == "inv_sqrt":
            # originally used for Transformer (in Attention is all you need)
            def lr_lambda(step):
                # return a multiplier instead of a learning rate
                if step == 0 and self.global_state.warmup_step == 0:
                    return 1.0
                else:
                    return (
                        1.0 / (step ** 0.5)
                        if step > self.global_state.warmup_step
                        else step / (self.global_state.warmup_step ** 1.5)
                    )

            scheduler = optim.lr_scheduler.LambdaLR(optimizer,
                                                    lr_lambda=lr_lambda)

        elif self.global_state.scheduler == "dev_perf":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=self.global_state.decay_rate,
                patience=self.global_state.patience,
                min_lr=self.global_state.lr_min,
            )

        elif self.global_state.scheduler == "constant":
            pass

        else:
            raise ValueError(
                f"scheduler type {self.global_state.scheduler} not recognized"
            )

        return scheduler

    #
    # STEP 3.3: Combine the two
    #
    def configure_optimizers(self):
        optimizer = self.build_optimizer()
        scheduler = self.build_scheduler(optimizer)
        return optimizer, scheduler

    ############################################################################
    #
    # STEP 4: How to train the transformer_model: first a dalaloader, the how to train
    #
    ############################################################################

    def train_dataloader(self):
        data_set = pd.read_pickle(
            f"{self.global_state.data_dir}/allData.pickle")

        data_set = data_set.fillna(0.0).values[:, 1:].astype(np.float32)
        data_set = torch.tensor(data_set)

        return DataLoader(
            IterableTimeSeries(self.global_state, data_set, mode="train"),
            batch_size=self.global_state.n_batch,
            num_workers=self.global_state.num_workers,
        )

    def training_step(self, batch, batch_nb, optimizer_idx=1):
        x, y = batch
        y_hat = self.forward(x, y)
        loss = self.criteria(y_hat, y)
        loss = loss.unsqueeze(dim=-1)
        return {"loss": loss}

    ############################################################################
    #
    # STEP 5: Then validate on a validation set dataloader (can be OPTIONAL)
    #
    ############################################################################

    def val_dataloader(self):
        data_set = pd.read_pickle(
            f"{self.global_state.data_dir}/allData.pickle")

        data_set = data_set.fillna(0.0).values[:, 1:].astype(np.float32)
        data_set = torch.tensor(data_set)

        return DataLoader(
            IterableTimeSeries(self.global_state, data_set, mode="val"),
            batch_size=self.global_state.n_batch,
            num_workers=self.global_state.num_workers,
        )

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x, y, self.transformer_model.mems)
        val_loss = self.criteria(y_hat, y)
        val_loss = val_loss.unsqueeze(dim=-1)
        return {"val_loss": val_loss}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        return {"avg_val_loss": avg_loss}

    # def validation_epoch_end(self, outputs):
    #     avg_loss = torch.stack([input['val_loss'] for input in outputs]).mean()
    #     tensorboard_logs = {'val_loss': avg_loss}
    #     return {'val_loss': avg_loss, 'log': tensorboard_logs}

    ############################################################################
    #
    # STEP 6: Finally test the transformer_model
    #
    ############################################################################

    def test_dataloader(self):
        data_set = pd.read_pickle(
            f"{self.global_state.data_dir}/allData.pickle")

        data_set = data_set.fillna(0.0).values[:, 1:].astype(np.float32)
        data_set = torch.tensor(data_set)

        return DataLoader(
            IterableTimeSeries(self.global_state, data_set, mode="test"),
            batch_size=self.global_state.n_batch,
            num_workers=self.global_state.num_workers,
        )

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return {"val_loss": loss}

    # def test_epoch_end(self, outputs):
    #     avg_loss = torch.stack([input['val_loss'] for input in outputs]).mean()
    #     tensorboard_logs = {'val_loss': avg_loss}
    #     return {'val_loss': avg_loss, 'log': tensorboard_logs}


################################################################################
#
# Checkpoint callback to save best 3 models
#
checkpoint_callback = ModelCheckpoint(
    filepath="./data/working/etf",
    save_top_k=3,
    verbose=True,
    monitor="avg_val_loss",
    mode="min",
)

# coding: utf-8

import os, inspect
from typing import *

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
    def __init__(self, global_state: GlobalState, data, mode="train",
                 debug=False):
        super(IterableTimeSeries, self).__init__()

        self.global_state = global_state
        self.data_type = mode

        if self.global_state.debug and (
                self.__class__.__name__ not in self.global_state.skip_debug):
            logging(f"{self.__class__.__name__}, "
                    f"{inspect.currentframe().f_code.co_name}: "
                    f" Creating dataloader for data set: {mode}")

        # In debug mode, only use about 2 epoch of input
        # TODO refactor to use exactly 2 epoch instead of 700 dates.
        if self.global_state.debug and (
                self.__class__.__name__ not in self.global_state.skip_debug):
            total_data_set_length = min(global_state.dataset_size, data.size(0))
        else:
            total_data_set_length = data.size(0)

        # The beginning of the data set is where 'train' starts
        # The end of the dataset is here we find the last testing data
        # We therefore start at 0
        # And end at total_data_set_length = n_samples + (n_model+1) + n_val + n_test
        # (a sample is n_model vectors for X and 1 vector for Y)
        # Final -1 is to reflect Python's 0-array convention
        self.n_samples = total_data_set_length - \
                         (global_state.n_model + 1) - \
                         global_state.n_val - \
                         global_state.n_test - \
                         1

        # Adjust the start of the dataset for training / val / test
        if mode == "train":
            start_index = 0
            end_index = (global_state.n_model + 1) + self.n_samples

        elif mode == "val":
            start_index = self.n_samples
            end_index = (global_state.n_model + 1) + self.n_samples + \
                        global_state.n_val

        elif mode == "test":
            start_index = self.n_samples + global_state.n_val
            end_index = (global_state.n_model + 1) + self.n_samples + \
                        global_state.n_val + \
                        global_state.n_test

        # This is the actual input on which to iterate
        self.data = data[start_index:end_index, :]

        if self.global_state.debug and (
                self.__class__.__name__ not in self.global_state.skip_debug):
            logging(f"{self.__class__.__name__}, "
                    f"{inspect.currentframe().f_code.co_name}: "
                    f" Dataset {self.data_type} - Start index: {start_index}")
            logging(f"{self.__class__.__name__}, "
                    f"{inspect.currentframe().f_code.co_name}: "
                    f" Dataset {self.data_type} - End index: {end_index}")
            logging(f"{self.__class__.__name__}, "
                    f"{inspect.currentframe().f_code.co_name}: "
                    f" Dataset {self.data_type} - data: {self.data.size()}")
            logging(f"{self.__class__.__name__}, "
                    f"{inspect.currentframe().f_code.co_name}: "
                    f" Dataset {self.data_type} - data set iterator"
                    f" length: {self.data.size()[0]}")
            logging(f"{self.__class__.__name__}, "
                    f"{inspect.currentframe().f_code.co_name}: "
                    f" Dataset {self.data_type} - calculated"
                    f" n_samples: {self.n_samples}")

        # d_series is the depth of a series (how many input points per dates)
        # n_series is the number of series (how many dates)
        self.n_series, self.d_series = data.size()

    def __getitem__(self, index):
        # An item is a tuple of:
        #   - a transformer_model input being, say, 60 dates of time series
        #   -  the following date as expected output
        if self.global_state.debug and (
                self.__class__.__name__ not in self.global_state.skip_debug):
            logging(f"{self.__class__.__name__}, "
                    f"{inspect.currentframe().f_code.co_name}: "
                    f" {self.data_type} \t item  no.: {index}")
            logging(f"{self.__class__.__name__}, "
                    f"{inspect.currentframe().f_code.co_name}: "
                    f"       x: from {index} to {index + self.global_state.n_model}")
            logging(f"{self.__class__.__name__}, "
                    f"{inspect.currentframe().f_code.co_name}: "
                    f"       y: at {index + self.global_state.n_model}")

        return (self.data[index: index + self.global_state.n_model, :],
                self.data[index + self.global_state.n_model, :])

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        if self.global_state.debug and (
                self.__class__.__name__ not in self.global_state.skip_debug):
            logging(f"{self.__class__.__name__}, "
                    f"{inspect.currentframe().f_code.co_name}: "
                    f" Call to __len__() on {self.data_type} returning"
                    f" self.data.size()[0] - (self.global_state.n_model + 1) ="
                    f" {self.data.size()[0] - (self.global_state.n_model + 1)}")
        return self.data.size()[0] - (self.global_state.n_model + 1)


################################################################################
##
## Lightning module of the transformer_model
##
class TransformerXL_Trainer(pl.LightningModule):
    def __init__(self, global_state: GlobalState):
        super(TransformerXL_Trainer, self).__init__()

        self.global_state = global_state

        if self.global_state.debug and (
                self.__class__.__name__ not in self.global_state.skip_debug):
            logging(f"")
            logging(f"")
            logging(f"########################################################"
                    f"########################################################")
            logging(f"########################################################"
                    f"########################################################")
            logging(f"")
            logging(f"    INITIALISING TRANSFORMER XL")
            logging(f"")
            logging(f"########################################################"
                    f"########################################################")
            logging(f"########################################################"
                    f"########################################################")
            logging(f"")

        self.transformer_model = Transformer_XL(
            n_layer=global_state.n_layer,
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
            debug=global_state.debug,
            skip_debug=global_state.skip_debug)

        self.loss_function = nn.MSELoss()

    ############################################################################
    #
    # STEP 1: Define the transformer_model
    #
    ############################################################################

    def forward(self, input: torch.FloatTensor, output: torch.FloatTensor,
                *mems):
        if self.global_state.debug and (
                self.__class__.__name__ not in self.global_state.skip_debug):
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
            optimizer = optim.Adam(params=self.transformer_model.parameters(),
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
                optimizer=optimizer,
                factor=self.global_state.decay_rate,
                patience=self.global_state.patience,
                min_lr=self.global_state.min_lr,
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

        # TODO: adding a scheduler throws errors. Check
        # scheduler = self.build_scheduler(optimizer=optimizer)
        # return optimizer, scheduler
        return optimizer

    ############################################################################
    #
    # STEP 4: How to train the transformer_model: first a dalaloader, the how to train
    #
    ############################################################################

    def train_dataloader(self):
        if self.global_state.debug and (
                self.__class__.__name__ not in self.global_state.skip_debug):
            logging(f"{self.__class__.__name__}, "
                    f"{inspect.currentframe().f_code.co_name}: "
                    f" Creating dataloader train")

        data_set = pd.read_pickle(
            f"{self.global_state.data_dir}/allData.pickle")

        data_set = data_set.fillna(0.0).values[:, 1:].astype(np.float32)
        data_set = torch.tensor(data_set)

        dataloader = DataLoader(
            IterableTimeSeries(self.global_state, data_set, mode="train"),
            batch_size=self.global_state.n_batch,
            num_workers=self.global_state.num_workers, drop_last=True
        )
        if self.global_state.debug and (
                self.__class__.__name__ not in self.global_state.skip_debug):
            logging(f"{self.__class__.__name__}, "
                    f"{inspect.currentframe().f_code.co_name}: "
                    f" Dataloader length: {len(dataloader)}")

        return dataloader

    def training_step(self, batch: List[torch.Tensor], batch_idx: int,
                      optimizer_idx: int = 1):
        # DIMS: batch = (x, y)
        # DIMS: x -> (n_batch, n_model, d_model)
        # DIMS: y -> (n_batch, d_model)
        x, y = batch

        if self.global_state.debug and (
                self.__class__.__name__ not in self.global_state.skip_debug):
            logging(f"{self.__class__.__name__}, "
                    f"{inspect.currentframe().f_code.co_name}: "
                    f" x = batch[0]: {batch[0].size()}")
            logging(f"{self.__class__.__name__}, "
                    f"{inspect.currentframe().f_code.co_name}: "
                    f" y = batch[1]: {batch[1].size()}")

        y_hat = self.forward(x, y)
        if self.global_state.debug and (
                self.__class__.__name__ not in self.global_state.skip_debug):
            logging(f"{self.__class__.__name__}, "
                    f"{inspect.currentframe().f_code.co_name}: "
                    f" y_hat['loss']: {y_hat['loss'].size()}")
            logging(f"{self.__class__.__name__}, "
                    f"{inspect.currentframe().f_code.co_name}: "
                    f" y_hat['layer_out']: {y_hat['layer_out'].size()}")
            logging(f"{self.__class__.__name__}, "
                    f"{inspect.currentframe().f_code.co_name}: "
                    f" y_hat['memory'][0]: {y_hat['memory'][0].size()}")

        loss = self.loss_function(y_hat['layer_out'][:, -1, :], y)
        if self.global_state.debug and (
                self.__class__.__name__ not in self.global_state.skip_debug):
            logging(f"{self.__class__.__name__}, "
                    f"{inspect.currentframe().f_code.co_name}: "
                    f" loss: {loss.size()}")

        loss = loss.unsqueeze(dim=-1)
        return {"loss": loss}

    ############################################################################
    #
    # STEP 5: Then validate on a validation set dataloader (can be OPTIONAL)
    #
    ############################################################################

    def val_dataloader(self):
        if self.global_state.debug and (
                self.__class__.__name__ not in self.global_state.skip_debug):
            logging(f"{self.__class__.__name__}, "
                    f"{inspect.currentframe().f_code.co_name}: "
                    f" Creating dataloader val")

        data_set = pd.read_pickle(
            f"{self.global_state.data_dir}/allData.pickle")

        data_set = data_set.fillna(0.0).values[:, 1:].astype(np.float32)
        data_set = torch.tensor(data_set)

        # Note that batches have size 1!
        dataloader = DataLoader(
            IterableTimeSeries(self.global_state, data_set, mode="val"),
            batch_size=1,
            num_workers=self.global_state.num_workers, drop_last=True
        )
        if self.global_state.debug and (
                self.__class__.__name__ not in self.global_state.skip_debug):
            logging(f"{self.__class__.__name__}, "
                    f"{inspect.currentframe().f_code.co_name}: "
                    f" Dataloader length: {len(dataloader)}")

        return dataloader

    def validation_step(self, batch, batch_nb):
        # DIMS: batch = (x, y)
        # DIMS: x -> (n_batch, n_model, d_model)
        # DIMS: y -> (n_batch, d_model)
        x, y = batch

        if self.global_state.debug and (
                self.__class__.__name__ not in self.global_state.skip_debug):
            logging(f"{self.__class__.__name__}, "
                    f"{inspect.currentframe().f_code.co_name}: "
                    f" x = batch[0]: {batch[0].size()}")
            logging(f"{self.__class__.__name__}, "
                    f"{inspect.currentframe().f_code.co_name}: "
                    f" y = batch[1]: {batch[1].size()}")

        y_hat = self.forward(x, y)
        if self.global_state.debug and (
                self.__class__.__name__ not in self.global_state.skip_debug):
            logging(f"{self.__class__.__name__}, "
                    f"{inspect.currentframe().f_code.co_name}: "
                    f" y_hat['loss']: {y_hat['loss'].size()}")
            logging(f"{self.__class__.__name__}, "
                    f"{inspect.currentframe().f_code.co_name}: "
                    f" y_hat['layer_out']: {y_hat['layer_out'].size()}")
            logging(f"{self.__class__.__name__}, "
                    f"{inspect.currentframe().f_code.co_name}: "
                    f" y_hat['memory'][0]: {y_hat['memory'][0].size()}")

        val_loss = self.loss_function(y_hat['layer_out'][:, -1, :], y)
        if self.global_state.debug and (
                self.__class__.__name__ not in self.global_state.skip_debug):
            logging(f"{self.__class__.__name__}, "
                    f"{inspect.currentframe().f_code.co_name}: "
                    f" loss: {val_loss.size()}")

        val_loss = val_loss.unsqueeze(dim=-1)
        return {"val_loss": val_loss}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        return {"val_loss": avg_loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([output['val_loss'] for output in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    ############################################################################
    #
    # STEP 6: Finally test the transformer_model
    #
    ############################################################################

    def test_dataloader(self):
        if self.global_state.debug and (
                self.__class__.__name__ not in self.global_state.skip_debug):
            logging(f"{self.__class__.__name__}, "
                    f"{inspect.currentframe().f_code.co_name}: "
                    f" Creating dataloader train")

        data_set = pd.read_pickle(
            f"{self.global_state.data_dir}/allData.pickle")

        data_set = data_set.fillna(0.0).values[:, 1:].astype(np.float32)
        data_set = torch.tensor(data_set)

        # Note that batches have size 1!
        dataloader = DataLoader(
            IterableTimeSeries(self.global_state, data_set, mode="test"),
            batch_size=1,
            num_workers=self.global_state.num_workers, drop_last=True
        )
        if self.global_state.debug and (
                self.__class__.__name__ not in self.global_state.skip_debug):
            logging(f"{self.__class__.__name__}, "
                    f"{inspect.currentframe().f_code.co_name}: "
                    f" Dataloader length: {len(dataloader)}")

        return dataloader

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return {"test_loss": loss}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([output['test_loss'] for output in outputs]).mean()
        tensorboard_logs = {'avg_test_loss': avg_loss}
        return {'avg_test_loss': avg_loss, 'log': tensorboard_logs}


################################################################################
#
# Checkpoint callback to save best 3 models
#
checkpoint_callback = ModelCheckpoint(
    filepath="./experiments/checkpoints/etf",
    save_top_k=3,
    verbose=True,
    monitor="avg_val_loss",
    mode="min",
)

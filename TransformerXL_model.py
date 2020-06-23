# coding: utf-8

import os

import itertools
from functools import partial
import warnings

import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from TransformerXL_modules import TransformerXL

from utils.exp_utils import create_exp_dir, logging
from utils.initialization import weights_init

from utils.argparsing import parser


class GlobalState():
    def __init__(self, data):
        self.data_dir = "./data/etf"
        self.data_pickle = "allData.pickle"

        # dimensionality of the model's hidden states'
        # depth of the model = no. of series = n_series
        self.d_model = data.shape[1]

        self.adapt_inp = False
        self.n_layer = 12

        # number of attention heads for each attention layer in the Transformer encoder
        self.n_head = 10

        # dimensionality of the model's heads
        self.d_head = 50

        # Dimensionality of the embeddings
        self.d_pos_embed = 20

        # model dimension. Must be even.
        self.n_model = 500

        self.d_inner = 1000
        self.n_train = 12
        self.n_val = 2
        self.n_test = 2

        # batch size"
        self.n_batch = 60
        self.batch_chunk = 1
        self.not_tied = False
        self.pre_lnorm = False
        self.dropout = 0.0
        self.dropatt = 0.0

        # parameter initializer to use.
        self.init = "normal"
        self.emb_init = "normal"
        self.init_range = 0.1
        self.emb_init_range = 0.01
        self.init_std = 0.02
        self.proj_init_std = 0.01

        # Choices: adam, sgd, adagrad
        self.optim = "adam"
        self.lr = 0.00025

        # momentum for sgd"
        self.mom = 0.0
        self.scheduler = "cosine"
        self.warmup_step = 0
        self.decay_rate = 0.5
        self.lr_min = 0.0
        self.clip = 0.25
        self.clip_nonemb = True
        self.eta_min = 0.0
        self.patience = 0

        # number of tokens to predict
        self.n_predict = 10
        self.eval_n_predict = 50

        # length of the extended context
        self.n_ext_ctx = 0
        self.n_mems = 0
        self.varlen = False
        self.same_length = True

        # use the same pos embeddings after n_clamp_after
        self.n_clamp_after = -1

        # random seed
        self.seed = 42
        self.max_step = 10000
        self.max_eval_steps = -1

        # use CUDA
        self.cuda = False
        self.multi_gpu = False
        self.gpu0_bsz = -1

        # choices: "O1", "O2", "O0"
        self.fp16 = None
        self.log_interval = 200

        # evaluation interval
        self.eval_interval = 4000

        # experiment directory
        self.work_dir = "experiments"
        self.restart = True

        # restart dir
        self.restart_dir = ""
        self.debug = False
        self.finetune_v2 = True
        self.finetune_v3 = True
        self.log_first_epochs = 0
        self.restart_from = None
        self.reset_lr = True

        # help="reset learning schedule to start"
        self.expand = None
        # help="Add layers to model throughout training
        # choices=["repeat", "reinit", "repeat_bottom", "reinit_bottom", "duplicate"],
        self.integration = ""
        # choices=["freeze", "reverse_distil_full",
        # "reverse_distil_partial"]
        self.integration_length = 0
        self.expansion_dict = {}
        self.widen = None  # choices=["reinit", "duplicate"]
        self.widen_dict = {}


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
        m.dropout_p = global_state.dropatt


def update_dropatt(global_state: GlobalState, m):
    if hasattr(m, "dropatt"):
        m.dropatt.p = global_state.dropatt
    if hasattr(m, "dropatt_p"):
        m.dropatt_p = global_state.dropatt


################################################################################
##
## Dataset class
##
class IterableTimeSeries(Dataset):
    def __init__(self, global_state: GlobalState, data,
                 mode="train"):

        # Keeps the size of a batch to start the test set
        self.n_batch = global_state.n_batch

        # In debug mode, only use about 2 epoch of data
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

        # This is the actual data on which to iterate
        self.data = actual_data[start_index:, :]

        # d_series is the depth of a series (how many data points per dates)
        # n_series is the number of series (how many dates)
        self.n_series, self.d_series = data.shape

        # Each training data point is a set of series to fill the model:
        # One date (d_series data points) for each entry to the model, that is
        # global_state.n_model
        self.n_model = global_state.n_model

    def __getitem__(self, index):
        return (
            self.data[index: index + self.n_model - 1, :],
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
class TransformerXL_Trainer(pl.LightningModule):
    def __init__(self, global_state: GlobalState):
        super(TransformerXL_Trainer, self).__init__()

        self.global_state = global_state

        self.model = TransformerXL(d_model=global_state.d_model,
                                   n_model=global_state.n_model,
                                   n_head=global_state.n_head,
                                   d_head=global_state.d_head,
                                   d_inner=global_state.d_inner,
                                   n_layer=global_state.n_layer,
                                   dropout=global_state.dropout,
                                   dropatt=global_state.dropatt,
                                   d_pos_embed=global_state.d_pos_embed,
                                   pre_lnorm=global_state.pre_lnorm,
                                   n_predict=global_state.n_predict,
                                   n_ext_ctx=global_state.n_ext_ctx,
                                   n_mems=global_state.n_mems,
                                   adapt_inp=global_state.adapt_inp,
                                   same_length=global_state.same_length,
                                   n_clamp_after=global_state.n_clamp_after)
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
    # Create clean data set with NaN -> 0 and remove the data index
    # WARNING Change when using market indices as well as returns (cannot fill
    # NaN indices with 0)
    def prepare_data(self) -> None:
        data_set = pd.read_pickle(
            f"{self.global_state.data_dir}/allData.pickle")

        data_set = data_set.fillna(0.0).values[:, 1:].astype(np.float64)
        data_set = torch.tensor(data_set)

        data_set.to_pickle(
            f"{self.global_state.data_dir}/allDataClean.pickle")
        return None

    ############################################################################
    #
    # STEP 3: Configure the optimizer (how to use the data in the model)
    #
    ############################################################################

    #
    # STEP 3.1: Build an optimizer
    #
    def build_optimizer(self, reload=False):
        if self.global_state.optim.lower() == "sgd":
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.global_state.lr,
                momentum=self.global_state.mom,
            )
        elif self.global_state.optim.lower() == "adam":
            optimizer = optim.Adam(self.model.parameters(),
                                   lr=self.global_state.lr)

        elif self.global_state.optim.lower() == "adagrad":
            optimizer = optim.Adagrad(
                self.model.parameters(), lr=self.global_state.lr
            )
        else:
            raise ValueError(
                f"optimizer type {self.global_state.optim} not recognized"
            )

        if reload:
            if self.global_state.restart_from is not None:
                optim_name = f"optimizer_{self.global_state.restart_from}.pt"
            else:
                optim_name = "optimizer.pt"

            optim_file_name = os.path.join(self.global_state.restart_dir,
                                           optim_name)
            logging(f"reloading {optim_file_name}")
            if os.path.exists(
                    os.path.join(self.global_state.restart_dir, optim_name)
            ):
                with open(
                        os.path.join(self.global_state.restart_dir, optim_name),
                        "rb") as optim_file:
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
    # STEP 4: How to train the model: first a dalaloader, the how to train
    #
    ############################################################################

    def train_dataloader(self):
        data_set = pd.read_pickle(
            f"{self.global_state.data_dir}/allDataClean.pickle")
        return DataLoader(
            IterableTimeSeries(self.global_state, data_set, mode="train"),
            batch_size=self.global_state.n_batch,
            num_workers=4 - 1,
        )

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
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
            f"{self.global_state.data_dir}/allDataClean.pickle")
        return DataLoader(
            IterableTimeSeries(self.global_state, data_set, mode="val"),
            batch_size=self.global_state.n_batch,
            num_workers=4 - 1,
        )

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        val_loss = self.criteria(y_hat, y)
        val_loss = val_loss.unsqueeze(dim=-1)
        return {"val_loss": val_loss}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        return {"avg_val_loss": avg_loss}

    # def validation_epoch_end(self, outputs):
    #     avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
    #     tensorboard_logs = {'val_loss': avg_loss}
    #     return {'val_loss': avg_loss, 'log': tensorboard_logs}

    ############################################################################
    #
    # STEP 6: Finally test the model
    #
    ############################################################################

    def test_dataloader(self):
        data_set = pd.read_pickle(
            f"{self.global_state.data_dir}/allDataClean.pickle")
        return DataLoader(
            IterableTimeSeries(self.global_state, data_set, mode="test"),
            batch_size=self.global_state.n_batch,
            num_workers=4 - 1,
        )

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return {'val_loss': loss}

    # def test_epoch_end(self, outputs):
    #     avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
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

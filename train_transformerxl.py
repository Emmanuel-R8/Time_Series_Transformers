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

from TransformerXL_model import Train_TransformerXL, GlobalState

from utils.exp_utils import create_exp_dir, logging
from utils.initialization import weights_init

from utils.argparsing import parser


################################################################################
##
## Main.
##
if __name__ == "__main__":
    args = parser.parse_args()

    # %% Load data to determiine the number of series
    dataDir = "./data/etf"
    data_set = pd.read_pickle(f"{dataDir}/allData.pickle")
    data_set.fillna(0, inplace=True)

    global_state = GlobalState(data=data_set)

    # Update the global parameters state for any value supplied as arguments
    gs_keys = global_state.__dict__.keys()

    for k, v in args.__dict__:
        if k in gs_keys:
            global_state.__dict__[k] = v

    # build a model
    train_transformerxl = Train_TransformerXL(global_state)

# coding: utf-8

from utils.argparsing import parser
import pandas as pd
import pytorch_lightning as pl

from TransformerXL_model import TransformerXL_Trainer, GlobalState

################################################################################
##
## Main.
##
if __name__ == "__main__":
    args = parser.parse_args()

    # %% Load input to determiine the number of series
    dataDir = "./input/etf"
    data_set = pd.read_pickle(f"{dataDir}/allData.pickle")
    data_set.fillna(0, inplace=True)

    global_state = GlobalState(data=data_set)

    # Update the global parameters state for any value supplied as arguments
    gs_keys = global_state.__dict__.keys()

    for k, v in args.__dict__:
        if k in gs_keys:
            global_state.__dict__[k] = v

    # build a transformer_model
    transformerxl_model = TransformerXL_Trainer(global_state)
    pl.seed_everything(global_state.seed)

    transformerxl_trainer = pl.Trainer()
    transformerxl_trainer.fit(transformerxl_model)


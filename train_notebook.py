# %%
import os
import numpy as np
import pandas as pd
import torch

import pytorch_lightning as pl
from pytorch_lightning.loggers import TestTubeLogger

from TransformerXL_model import TransformerXL_Trainer, GlobalState

# %% Load data to determiine the number of series
data_set_name = "etf"

data_set = pd.read_pickle(f"data/{data_set_name}/allData.pickle")

data_set = data_set.fillna(0.0).values[:, 1:].astype(np.float64)
data_set  = torch.tensor(data_set)

# %%
global_state = GlobalState(data=data_set)

# %%
# Create a new model to be trained
transformerxl_model = TransformerXL_Trainer(global_state)

# %%
# Logging backend
TB_logger = TestTubeLogger(
    save_dir=os.path.join(os.getcwd(), global_state.work_dir, data_set_name),
    name='transformerxl_trainer_logs'
)

#%%
# Actual Trainer() method
transformerxl_trainer = pl.Trainer(logger=TB_logger)

# %%
pl.seed_everything(global_state.seed)
transformerxl_trainer.fit(transformerxl_model)

# %%

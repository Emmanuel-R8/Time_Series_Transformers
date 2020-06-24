# %%
import os
import numpy as np
import pandas as pd
import torch

import pytorch_lightning as pl
from pytorch_lightning.loggers import TestTubeLogger

from TransformerXL_model import TransformerXL_Trainer, GlobalState

# %% Load input
data_set_name = "etf"
data_set = pd.read_pickle(f"data/{data_set_name}/allData.pickle")

# Fill NA's with 0.0 and convert to tensor (although probably not necessary)
data_set = data_set.fillna(0.0).values[:, 1:].astype(np.float64)
data_set = torch.tensor(data_set)

# %%
global_state = GlobalState(data=data_set, debug=True)

# %%
# Create a new transformer_model to be trained
transformerxl_model = TransformerXL_Trainer(global_state)

# %%
# Logging backend
TB_logger = TestTubeLogger(
    save_dir=os.path.join(os.getcwd(), global_state.work_dir, data_set_name),
    name='transformerxl_trainer_logs'
)

# %%
# Actual Trainer() method
# Avoids preliminary runs of validation before training
transformerxl_trainer = pl.Trainer(logger=TB_logger, num_sanity_val_steps=0)

# %%
pl.seed_everything(global_state.seed)
transformerxl_trainer.fit(transformerxl_model)

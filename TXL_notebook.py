###############################################################
#
# GLOBAL CONSTANTS
#

NEPTUNE_USER_NAME = 'emmanuel-r8'
NEPTUNE_PROJECT_NAME = 'TS_TXL'

# If we run on Google Colab, we don't have exactly the same parameteres"
USE_COLAB = False

# %%
import os
import numpy as np
import pandas as pd
import torch

# Import colab TPU libraries
if USE_COLAB:
    assert os.environ[
        'COLAB_TPU_ADDR'], 'Make sure to select TPU from Edit > Notebook settings > Hardware accelerator'
    os.chdir("/content/drive/My Drive/Colab Notebooks/Time_Series_Transformers")

    # FROM https://colab.research.google.com/github/pytorch/xla/blob/master/contrib/colab/getting-started.ipynb#scrollTo=sPJVqAKyml5W
    VERSION = "20200325"  # @param [\"1.5\" , \"20200325\", \"nightly\"]
    os.system(
        "curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py")
    os.system(f"python pytorch-xla-env-setup.py --version {VERSION}")
    os.system(
        "pip install pytorch-lightning test-tube neptune-client torch-xla")

    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp

# Now import usual libraries
import pytorch_lightning as pl
import pytorch_lightning.profiler as plprof

from pytorch_lightning.loggers import NeptuneLogger

from model import TransformerXL_Trainer
from utils.utils import GlobalState

# %% Load input
data_set_name = "etf"
data_set = pd.read_pickle(f"data/{data_set_name}/allData.pickle")

# Fill NA's with 0.0 and convert to tensor (although probably not necessary)
data_set = data_set.fillna(0.0).values[:, 1:].astype(np.float64)
data_set = torch.tensor(data_set)

# %%
global_state = GlobalState(data=data_set, debug=False)
global_state.num_workers = 4
global_state.dataset_size = 1_500
global_state.n_layer = 3
global_state.n_head = 4
global_state.d_head = 8
global_state.n_model = 20
global_state.d_pos_enc = 20
global_state.d_FF_inner = 50
global_state.n_val = 8
global_state.n_test = 16
global_state.n_batch = 32
global_state.dropout = 0.2
global_state.dropout_attn = 0.2
global_state.n_predict = 8
global_state.n_mems = 16
global_state.max_epochs = 3
global_state.profiler = plprof.AdvancedProfiler(
    output_filename=f"experiments/profiles/{global_state.data_set}.profile.txt")

# %%
# Create a new transformer_model to be trained
transformerxl_model = TransformerXL_Trainer(global_state)

# %%
# Logging backend
# TB_logger = TestTubeLogger(
#     save_dir=os.path.join(os.getcwd(), global_state.work_dir, data_set_name),
#     name='transformerxl_trainer_logs'
# )
neptune_logger = NeptuneLogger(
    offline_mode=True,
    project_name=f'{NEPTUNE_USER_NAME}/{NEPTUNE_PROJECT_NAME}',
    experiment_name='small_1500',
    params={'max_epochs': 200},
    tags=['pytorch-lightning', 'time_series', 'TXL'])

# %%
# Actual Trainer() method
# Avoids preliminary runs of validation before training
transformerxl_trainer = pl.Trainer(max_epochs=global_state.max_epochs,
                                   logger=neptune_logger,
                                   num_sanity_val_steps=0,
                                   profiler=global_state.profiler,
                                   distributed_backend='ddp')

# %%
pl.seed_everything(global_state.seed)
transformerxl_trainer.fit(transformerxl_model)

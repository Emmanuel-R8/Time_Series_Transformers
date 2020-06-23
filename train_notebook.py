# %%
import pandas as pd
from TransformerXL_model import Train_TransformerXL, GlobalState

# %% Load data to determiine the number of series
dataDir = "./data/etf"
data_set = pd.read_pickle(f"{dataDir}/allData.pickle")
data_set.fillna(0, inplace=True)

# %%
global_state = GlobalState(data=data_set)

# %%
train_transformerxl = Train_TransformerXL(global_state)

# %%

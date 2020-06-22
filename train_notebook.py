# %%
import pandas as pd
import train_transformerxl as txl

# %% Load data to determiine the number of series
dataDir = "./data/etf"
data_set = pd.read_pickle(f"{dataDir}/allData.pickle")
data_set.fillna(0, inplace=True)

# %%

globalState = {
    "datadir"           : "./data/etf",
    "d_model"           : data_set.shape[1],
    # depth of the model = no. of series = n_series
    "adapt_inp"         : False,
    "n_layer"           : 12,
    "n_head"            : 10,
    "d_head"            : 50,
    "d_embed"           : -1,
    "n_model"           : 500,  # model dimension. Must be even.
    "d_inner"           : 1000,
    "n_train"           : 12,
    "n_val"             : 2,
    "n_test"            : 2,
    "n_batch"           : 60,  # batch size"
    "batch_chunk"       : 1,
    "not_tied"          : False,
    "pre_lnorm"         : False,
    "dropout"           : 0.0,
    "dropatt"           : 0.0,
    "init"              : "normal",  # parameter initializer to use.
    "emb_init"          : "normal",
    "init_range"        : 0.1,
    "emb_init_range"    : 0.01,
    "init_std"          : 0.02,
    "proj_init_std"     : 0.01,
    "optim"             : "adam",  # adam, sgd, adagrad
    "lr"                : 0.00025,
    "mom"               : 0.0,  # momentum for sgd"
    "scheduler"         : "cosine",
    "warmup_step"       : 0,
    "decay_rate"        : 0.5,
    "lr_min"            : 0.0,
    "clip"              : 0.25,
    "clip_nonemb"       : True,
    "eta_min"           : 0.0,
    "patience"          : 0,
    "tgt_len"           : 70,  # help="number of tokens to predict"
    "eval_tgt_len"      : 50,
    "ext_len"           : 0,  # help="length of the extended context"
    "mem_len"           : 0,
    "varlen"            : False,
    "same_length"       : True,
    "clamp_len"         : -1,
    "seed"              : 42,  # help="random seed"
    "max_step"          : 10000,
    "max_eval_steps"    : -1,
    "cuda"              : False,  # help="use CUDA"
    "multi_gpu"         : False,
    "gpu0_bsz"          : -1,
    "fp16"              : None,  # choices=["O1", "O2", "O0"],
    "log-interval"      : 200,
    "eval-interval"     : 4000,  # help="evaluation interval"
    "work_dir"          : "TXL_TS",  # help="experiment directory."
    "restart"           : True,
    "restart_dir"       : "",  # help="restart dir")
    "debug"             : True,
    "finetune_v2"       : True,
    "finetune_v3"       : True,
    "log_first_epochs"  : 0,
    "restart_from"      : None,
    "reset_lr"          : True,  # help="reset learning schedule to start"
    "expand"            : None,  # help="Add layers to model throughout training
    # choices=["repeat", "reinit", "repeat_bottom", "reinit_bottom", "duplicate"],
    "integration"       : "",  # choices=["freeze", "reverse_distil_full",
    # "reverse_distil_partial"]
    "integration_length": 0,
    "expansion_dict"    : {},
    "widen"             : None,  # choices=["reinit", "duplicate"],
    "widen_dict"        : {},
}

# %%
train_transformerxl = txl.Train_TransformerXL(globalState)

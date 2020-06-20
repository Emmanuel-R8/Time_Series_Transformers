import argparse


class StoreDictKeyPair(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = {}
        for kv in values.split(","):
            k, v = kv.split("=")
            my_dict[k] = v
        setattr(namespace, self.dest, my_dict)


parser = argparse.ArgumentParser(
    description="PyTorch Lightning TransformerXL Model")
parser.add_argument(
    "--datadir", type=str, default="./data/etf",
    help="location of the data time_series"
)
parser.add_argument("--n_layer", type=int, default=12,
                    help="number of total layers")
parser.add_argument("--n_head", type=int, default=10, help="number of heads")
parser.add_argument("--d_head", type=int, default=50, help="head dimension")
parser.add_argument("--d_embed", type=int, default=-1,
                    help="embedding dimension")
parser.add_argument(
    "--n_model", type=int, default=500, help="model dimension. Must be even."
)
parser.add_argument("--d_inner", type=int, default=1000,
                    help="inner dimension in FF")
# parser.add_argument(
#     "--time_unit", type=str, default='month',
#     choices=["day", "week", "month", "quarter"],
#     help="Unit in which the all sets are measured. For example, n_batch 3 and "
#          "time_unit='month' means that the batch are all the days within a 3 "
#          "month-window. "
# )
parser.add_argument(
    "--n_train", type=int, default=12,
    help="Size of the training set expressed in time_unit. Default: 12 months")
parser.add_argument(
    "--n_val", type=int, default=2,
    help="Size of the validation set expressed in time_unit. Default: 2 months")
parser.add_argument(
    "--n_test", type=int, default=2,
    help="Size of the test set expressed in time_unit. Default: 2 months")
parser.add_argument("--n_batch", type=int, default=60, help="batch size")
parser.add_argument(
    "--batch_chunk", type=int, default=1,
    help="split batch into chunks to save memory"
)
parser.add_argument(
    "--not_tied",
    action="store_true",
    help="do not tie the word embedding and sigmoid weights",
)
parser.add_argument(
    "--div_val",
    type=int,
    default=1,
    help="divident value for adapative input and sigmoid",
)
parser.add_argument(
    "--pre_lnorm",
    action="store_false",
    help="apply LayerNorm to the input instead of the output",
)
parser.add_argument("--dropout", type=float, default=0.0,
                    help="global dropout rate")
parser.add_argument(
    "--dropatt", type=float, default=0.0,
    help="attention probability dropout rate"
)

parser.add_argument(
    "--init", default="normal", type=str, help="parameter initializer to use."
)
parser.add_argument(
    "--emb_init", default="normal", type=str,
    help="parameter initializer to use."
)
parser.add_argument(
    "--init_range",
    type=float,
    default=0.1,
    help="parameters initialized by U(-init_range, init_range)",
)
parser.add_argument(
    "--emb_init_range",
    type=float,
    default=0.01,
    help="parameters initialized by U(-init_range, init_range)",
)
parser.add_argument(
    "--init_std",
    type=float,
    default=0.02,
    help="parameters initialized by N(0, init_std)",
)
parser.add_argument(
    "--proj_init_std",
    type=float,
    default=0.01,
    help="parameters initialized by N(0, init_std)",
)

parser.add_argument(
    "--optim",
    default="adam",
    type=str,
    choices=["adam", "sgd", "adagrad"],
    help="optimizer to use.",
)
parser.add_argument(
    "--lr",
    type=float,
    default=0.00025,
    help="initial learning rate (0.00025|5 for adam|sgd)",
)
parser.add_argument("--mom", type=float, default=0.0, help="momentum for sgd")
parser.add_argument(
    "--scheduler",
    default="cosine",
    type=str,
    choices=["cosine", "inv_sqrt", "dev_perf", "constant"],
    help="lr scheduler to use.",
)
parser.add_argument("--warmup_step", type=int, default=0,
                    help="upper epoch limit")
parser.add_argument(
    "--decay_rate",
    type=float,
    default=0.5,
    help="decay factor when ReduceLROnPlateau is used",
)
parser.add_argument(
    "--lr_min", type=float, default=0.0,
    help="minimum learning rate during annealing"
)
parser.add_argument("--clip", type=float, default=0.25,
                    help="gradient clipping")
parser.add_argument(
    "--clip_nonemb",
    action="store_true",
    help="only clip the gradient of non-embedding params",
)
parser.add_argument(
    "--eta_min", type=float, default=0.0,
    help="min learning rate for cosine scheduler"
)
parser.add_argument("--patience", type=int, default=0, help="patience")

parser.add_argument(
    "--tgt_len", type=int, default=70, help="number of tokens to predict"
)
parser.add_argument(
    "--eval_tgt_len",
    type=int,
    default=50,
    help="number of tokens to predict for evaluation",
)
parser.add_argument(
    "--ext_len", type=int, default=0, help="length of the extended context"
)
parser.add_argument(
    "--mem_len", type=int, default=0,
    help="length of the retained previous heads"
)
parser.add_argument("--varlen", action="store_false",
                    help="use variable length")
parser.add_argument(
    "--same_length", action="store_true",
    help="use the same attn length for all tokens"
)
parser.add_argument(
    "--clamp_len",
    type=int,
    default=-1,
    help="use the same pos embeddings after clamp_len",
)

parser.add_argument("--seed", type=int, default=1111, help="random seed")
parser.add_argument("--max_step", type=int, default=10000,
                    help="upper limit of number of steps")
parser.add_argument("--max_eval_steps", type=int, default=-1,
                    help="max eval steps")
parser.add_argument("--cuda", action="store_false", help="use CUDA")
parser.add_argument("--multi_gpu", action="store_false",
                    help="use multiple GPU")
parser.add_argument("--gpu0_bsz", type=int, default=-1,
                    help="batch size on gpu 0")
parser.add_argument(
    "--fp16",
    type=str,
    default=None,
    choices=["O1", "O2", "O0"],
    help="activate amp training with the chosen mode",
)

parser.add_argument("--log-interval", type=int, default=200,
                    help="report interval")
parser.add_argument(
    "--eval-interval", type=int, default=4000, help="evaluation interval"
)
parser.add_argument(
    "--work_dir", default="LM-TFM", type=str, help="experiment directory."
)
parser.add_argument(
    "--restart", action="store_true",
    help="restart training from the saved checkpoint"
)
parser.add_argument("--restart_dir", type=str, default="", help="restart dir")
parser.add_argument(
    "--debug", action="store_true",
    help="run in debug mode (do not create exp dir)"
)
parser.add_argument("--finetune_v2", action="store_true", help="finetune v2")
parser.add_argument("--finetune_v3", action="store_true", help="finetune v3")
parser.add_argument(
    "--log_first_epochs", type=int, default=0,
    help="number of first epochs to log"
)
parser.add_argument(
    "--restart_from",
    type=int,
    default=None,
    help="restart from specific epoch checkpoint",
)
parser.add_argument(
    "--reset_lr", action="store_true", help="reset learning schedule to start"
)
parser.add_argument(
    "--expand",
    type=str,
    default=None,
    help="Add layers to model throughout training.",
    choices=["repeat", "reinit", "repeat_bottom", "reinit_bottom", "duplicate"],
)
parser.add_argument(
    "--integration",
    type=str,
    default="",
    help="Learning tricks post-expansion",
    choices=["freeze", "reverse_distil_full", "reverse_distil_partial"],
)
parser.add_argument(
    "--integration_length",
    type=int,
    default=0,
    help="Number of batches for reverse distillation or freezing",
)
parser.add_argument(
    "--expansion_dict",
    action=StoreDictKeyPair,
    metavar="KEY1=VAL1,KEY2=VAL2...",
    default={},
    help='Pass a dictionary formatted "KEYi=VALi,KEYj=VALj..."'
         " to indicate how many layers should be added at which epochs",
)
parser.add_argument(
    "--widen",
    type=str,
    default=None,
    help="widen the model throughout training.",
    choices=["reinit", "duplicate"],
)
parser.add_argument(
    "--widen_dict",
    action=StoreDictKeyPair,
    metavar="KEY1=VAL1,KEY2=VAL2...",
    default={},
    help='Pass a dictionary formatted "KEYi=VALi,KEYj=VALj..."'
         " to indicate what the ratio of widening should be at which epochs",
)

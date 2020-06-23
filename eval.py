# coding: utf-8
import argparse
import time
import math
import os

import torch

from data_utils import get_time_series
from utils.exp_utils import get_logger

parser = argparse.ArgumentParser(
    description="PyTorch Transformer Language Model"
)
parser.add_argument(
    "--data",
    type=str,
    default="../data/etf/allData.pickle",
    help="location of the data file",
)
parser.add_argument(
    "--split",
    type=str,
    default="all",
    choices=["all", "valid", "test"],
    help="which split to evaluate",
)
parser.add_argument("--n_batch", type=int, default=10, help="batch size")
parser.add_argument(
    "--n_predict", type=int, default=5, help="number of tokens to predict"
)
parser.add_argument(
    "--n_ext_ctx", type=int, default=0, help="length of the extended context"
)
parser.add_argument(
    "--n_mems",
    type=int,
    default=0,
    help="length of the retained previous heads",
)
parser.add_argument(
    "--n_clamp_after", type=int, default=-1,
    help="max positional embedding index"
)
parser.add_argument("--cuda", action="store_true", help="use CUDA")
parser.add_argument(
    "--work_dir", type=str, required=True, help="path to the work_dir"
)
parser.add_argument(
    "--no_log", action="store_true", help="do not log the eval result"
)
parser.add_argument(
    "--same_length",
    action="store_true",
    help="set same length attention with masking",
)
args = parser.parse_args()
assert args.n_ext_ctx >= 0, "extended context length must be non-negative"

device = torch.device("cuda" if args.cuda else "cpu")

# Get logger
logging = get_logger(
    os.path.join(args.work_dir, "log.txt"), log_=not args.no_log
)

# Load dataset
corpus = get_time_series(args.data, args.dataset)
nseries = len(corpus.vocab)

va_iter = corpus.get_iterator(
    "valid", args.batch_size, args.n_predict, device=device,
    n_ext_ctx=args.n_ext_ctx
)
te_iter = corpus.get_iterator(
    "test", args.batch_size, args.n_predict, device=device,
    n_ext_ctx=args.n_ext_ctx
)

# Load the best saved model.
with open(os.path.join(args.work_dir, "model.pt"), "rb") as f:
    model = torch.load(f)
model = model.to(device)

logging(
    "Evaluating with n_batch {} n_predict {} n_ext_ctx {} n_mems {} n_clamp_after {}".format(
        args.batch_size,
        args.n_predict,
        args.n_ext_ctx,
        args.n_mems,
        args.n_clamp_after,
    )
)

model.reset_length(args.n_predict, args.n_ext_ctx, args.n_mems)
if args.n_clamp_after > 0:
    model.n_clamp_after = args.n_clamp_after
if args.same_length:
    model.same_length = True


###############################################################################
# Evaluation code
###############################################################################


def evaluate(eval_iter):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_len, total_loss = 0, 0.0
    start_time = time.time()
    with torch.no_grad():
        mems = tuple()
        for idx, (data, target, seq_len) in enumerate(eval_iter):
            ret = model(data, target, *mems)
            loss, mems = ret[0], ret[1:]
            loss = loss.mean()
            total_loss += seq_len * loss.item()
            total_len += seq_len
        total_time = time.time() - start_time
    logging(
        "Time : {:.2f}s, {:.2f}ms/segment".format(
            total_time, 1000 * total_time / (idx + 1)
        )
    )
    return total_loss / total_len


# Run on test data.
if args.split == "all":
    test_loss = evaluate(te_iter)
    valid_loss = evaluate(va_iter)
elif args.split == "valid":
    valid_loss = evaluate(va_iter)
    test_loss = None
elif args.split == "test":
    test_loss = evaluate(te_iter)
    valid_loss = None


def format_log(loss, split):
    log_str = "| {0} loss {1:5.2f} | {0} bpc {2:9.5f} ".format(
        split, loss, loss / math.log(2)
    )

    return log_str


log_str = ""
if valid_loss is not None:
    log_str += format_log(valid_loss, "valid")
if test_loss is not None:
    log_str += format_log(test_loss, "test")

logging("=" * 100)
logging(log_str)
logging("=" * 100)

import os

import numpy as np
import torch


class OrderedIterator():
    def __init__(self, data: torch.LongTensor, d_batch: int = 10,
                 n_batch_per_test: int = 1,
                 device="cpu",
                 n_ext_ctx=None):
        """
            input -- LongTensor -- the LongTensor is strictly ordered
        """
        self.d_batch = d_batch
        self.bptt = n_batch_per_test
        self.n_ext_ctx = 0 if n_ext_ctx is None else n_ext_ctx

        self.device = device

        # Work out how cleanly we can divide the dataset into n_batch parts.
        self.n_dates = data.size(0)  # number of training dates
        self.n_batch = self.n_dates // d_batch

        # Trim off any extra elements that wouldn't cleanly fit (stub).
        # Remove the very beginning of the time series - keep the most recent
        data = data.narrow(0, self.n_dates % d_batch, data.size(0))

        # Evenly divide the input across the n_batch batches.
        self.data = data.view(d_batch, -1).t().contiguous().to(device)

        # Number of mini-batches
        self.n_batch = (self.n_batch + self.bptt - 1) // self.bptt

    def get_batch(self, i, bptt=None):
        if bptt is None:
            bptt = self.bptt

        seq_len = min(bptt, self.data.size(0) - 1 - i)

        end_idx = i + seq_len
        beg_idx = max(0, i - self.n_ext_ctx)

        data = self.data[beg_idx:end_idx]
        target = self.data[i + 1: i + 1 + seq_len]

        return data, target, seq_len

    def get_fixlen_iter(self, start=0):
        for i in range(start, self.data.size(0) - 1, self.bptt):
            yield self.get_batch(i)

    def get_varlen_iter(self, start=0, std=5, min_len=5, max_deviation=3):
        max_len = self.bptt + max_deviation * std
        i = start
        while True:
            bptt = self.bptt if np.random.random() < 0.95 else self.bptt / 2.0
            bptt = min(max_len, max(min_len, int(np.random.normal(bptt, std))))
            data, target, seq_len = self.get_batch(i, bptt)
            i += seq_len
            yield data, target, seq_len
            if i >= self.data.size(0) - 2:
                break

    def __iter__(self):
        return self.get_fixlen_iter()


def get_time_series(datadir, dataset):
    print("Loading cached dataset for: ")
    print(f"   data_dir: {datadir}")
    print(f"   dataset: {dataset}")

    fn = os.path.join(datadir, "cache.pt")
    print(f"   cache path: {fn}")

    if os.path.exists(fn):
        times_series_corpus = torch.load(fn)
    else:
        print("Producing dataset {}...".format(dataset))
        kwargs = {}

        times_series_corpus = TimeSeries(datadir, dataset, **kwargs)
        torch.save(times_series_corpus, fn)

    return times_series_corpus


class GlobalState:
    def __init__(self, data, debug=False):

        #######################################################
        #
        # Directories
        #
        self.data_dir = "./data/etf"
        self.data_pickle = "allData.pickle"
        self.dataset_size = 1_500 if debug == True else 1_000_000_000

        # experiment directory
        self.work_dir = "./experiments/logs"

        #######################################################
        #
        # Debugging
        #
        self.debug = debug
        # If debug is on, do not debug those functions
        self.skip_debug = ['']

        #######################################################
        #
        # Dimensions
        #

        # dimensionality of the transformer_model's hidden states'
        # depth of the transformer_model = no. of series = n_series
        self.d_model = data.shape[1]

        self.adapt_inp = False
        self.n_layer = 3 if debug == True else 8

        # number of attention heads for each attention layer in the Transformer
        # encoder
        self.n_head = 4 if debug == True else 16

        # dimensionality of the transformer_model's heads
        self.d_head = 8 if debug == True else 32

        # dimensionality of the hidden states
        self.d_hidden = self.d_model

        # Dimensionality of the embeddings - must be EVEN
        self.d_pos_enc = 12 if debug == True else self.d_model // 2

        # transformer_model dimension. Must be even.
        self.n_model = 13 if debug == True else 60

        self.d_FF_inner = 4 if debug == True else 16

        # TODO: Check that n_train is actually used
        self.n_train = 12
        self.n_val = 6 if debug == True else 12
        self.n_test = 9 if debug == True else 12

        # batch size"
        self.n_batch = 19 if debug == True else 64
        self.batch_chunk = 1
        self.not_tied = False
        self.pre_lnorm = False
        self.dropout = 0.0
        self.dropout_attn = 0.0

        # When debugging, dataloaders will run in the main process
        self.num_workers = 4 if debug == True else 4

        # number of tokens to predict
        self.n_predict = 3 if debug == True else 10
        self.eval_n_predict = 5 if debug == True else 20

        # length of the extended context
        self.n_ext_ctx = 2 if debug == True else 16
        self.n_mems = 2 if debug == True else 64
        self.varlen = False
        self.same_length = True

        # use the same pos embeddings after n_clamp_after
        self.n_clamp_after = -1

        # parameter initializer to use.
        self.init = "normal"
        self.emb_init = "normal"
        self.init_range = 0.1
        self.emb_init_range = 0.01
        self.init_std = 0.02
        self.proj_init_std = 0.01

        #######################################################
        #
        # Running parameters
        #
        self.max_epochs = 100 if debug == True else 1_000
        # Optimizer / Scheduler
        # Choices: adam, sgd, adagrad
        self.optim = "adam"
        self.lr = 0.00025

        # Choices: cosine, inv_sqrt, dev_perf, constant
        self.scheduler = "dev_perf"
        self.warmup_step = 0
        self.decay_rate = 0.5
        self.min_lr = 0.0
        self.clip = 0.25
        self.clip_nonemb = True
        self.eta_min = 0.0
        self.patience = 0

        # momentum for sgd
        self.mom = 0.0

        # random seed
        self.seed = 42
        self.max_step = 4 if debug == True else 512
        self.max_eval_steps = -1

        self.log_interval = 10 if debug == True else 200

        # evaluation interval
        self.eval_interval = 20 if debug == True else 200

        # Restart
        self.restart_dir = ""
        self.restart = True
        self.restart_from = None

        self.finetune_v2 = True
        self.finetune_v3 = True
        self.log_first_epochs = 0
        self.reset_lr = True

        # TODO reset learning schedule to start
        self.expand = None

        # TODO Add layers to transformer_model throughout training
        # choices: "repeat", "reinit", "repeat_bottom", "reinit_bottom", "duplicate"
        self.integration = ""

        # choices=["freeze", "reverse_distil_full",
        # "reverse_distil_partial"]
        self.integration_length = 0
        self.expansion_dict = {}

        # TODO Add layers to transformer_model throughout training
        # choices: "reinit", "duplicate"
        self.widen = None
        self.widen_dict = {}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="unit test")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./input/etf",
        help="location of the dataset",
    )
    args = parser.parse_args()

    time_series = get_time_series(args.datadir, args.dataset)
    print("Vocab size : {}".format(len(time_series.vocab.idx2sym)))

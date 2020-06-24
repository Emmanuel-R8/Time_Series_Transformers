import os
import sys
import glob

from collections import Counter, OrderedDict
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

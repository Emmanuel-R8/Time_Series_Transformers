import os
import sys
import glob

from collections import Counter, OrderedDict
import numpy as np
import torch

from utils.vocabulary import Vocab


class OrderedIterator():
    def __init__(self, data, bsz, bptt, device="cpu", ext_len=None):
        """
            data -- LongTensor -- the LongTensor is strictly ordered
        """
        self.bsz = bsz
        self.bptt = bptt
        self.ext_len = ext_len if ext_len is not None else 0

        self.device = device

        # Work out how cleanly we can divide the dataset into bsz parts.
        self.n_step = data.size(0) // bsz

        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, self.n_step * bsz)

        # Evenly divide the data across the bsz batches.
        self.data = data.view(bsz, -1).t().contiguous().to(device)

        # Number of mini-batches
        self.n_batch = (self.n_step + self.bptt - 1) // self.bptt

    def get_batch(self, i, bptt=None):
        if bptt is None:
            bptt = self.bptt

        seq_len = min(bptt, self.data.size(0) - 1 - i)

        end_idx = i + seq_len
        beg_idx = max(0, i - self.ext_len)

        data = self.data[beg_idx:end_idx]
        target = self.data[i + 1 : i + 1 + seq_len]

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


class TimeSeries(object):
    def __init__(self, path, dataset, *args, **kwargs):
        self.dataset = dataset
        self.vocab = Vocab(*args, **kwargs)

        self.vocab.count_file(os.path.join(path, "train.txt"))
        self.vocab.count_file(os.path.join(path, "valid.txt"))
        self.vocab.count_file(os.path.join(path, "test.txt"))

        self.vocab.build_vocab()

        self.train = self.vocab.encode_file(
            os.path.join(path, "train.txt"), ordered=True, add_eos=False
        )
        self.valid = self.vocab.encode_file(
            os.path.join(path, "valid.txt"), ordered=True, add_eos=False
        )
        self.test = self.vocab.encode_file(
            os.path.join(path, "test.txt"), ordered=True, add_eos=False
        )

    def get_iterator(self, split, *args, **kwargs):
        if split == "train":
            data_iter = OrderedIterator(self.train, *args, **kwargs)

        elif split in ["valid", "test"]:
            data = self.valid if split == "valid" else self.test
            data_iter = OrderedIterator(data, *args, **kwargs)

        return data_iter


def get_time_series(datadir, dataset):
    print("Loading cached dataset for: ")
    print(f"   datadir: {datadir}")
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
        "--datadir",
        type=str,
        default="./data/etf",
        help="location of the dataset",
    )
    args = parser.parse_args()

    corpus = get_time_series(args.datadir, args.dataset)
    print("Vocab size : {}".format(len(corpus.vocab.idx2sym)))

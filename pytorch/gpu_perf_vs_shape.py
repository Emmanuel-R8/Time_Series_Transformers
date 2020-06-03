# coding: utf-8
import argparse
import time
from itertools import product
from functools import partial
import gc
import json

import torch
import torch.nn as nn
from apex import amp

from data_utils import get_lm_corpus
from mem_transformer import MemTransformerLM
from utils.initialization import weights_init
from train import parallelize_model, build_optimizer, build_scheduler
from utils.torch_utils import non_emb_param_count, openai_compute


class DotDict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def head_repartition_rule(d_model):
    if d_model > 256:
        d_head = 64
        n_head = d_model // d_head
    elif d_model > 128:
        n_head = 8
        d_head = d_model // n_head
    elif d_model > 64:
        n_head = 4
        d_head = d_model // n_head
    elif d_model > 16:
        n_head = 2
        d_head = d_model // n_head
    else:
        n_head = 1
        d_head = d_model
    return n_head, d_head


def benchmark(model, optimizers, schedulers):
    # Turn on training mode which enables dropout.
    if isinstance(model, nn.DataParallel):
        parent_model = model.module
    else:
        parent_model = model
    optimizer, optimizer_sparse = optimizers
    scheduler, scheduler_sparse = schedulers
    train_step = 0
    train_losses = []
    model.train()
    if default_args.batch_chunk > 1:
        mems = [tuple() for _ in range(default_args.batch_chunk)]
    else:
        mems = tuple()
    train_iter = tr_iter.get_varlen_iter() if default_args.varlen else tr_iter
    start_time = time.time()
    for batch, (data, target, seq_len) in enumerate(train_iter):
        model.zero_grad()
        if default_args.batch_chunk > 1:
            data_chunks = torch.chunk(data, default_args.batch_chunk, 1)
            target_chunks = torch.chunk(target, default_args.batch_chunk, 1)
            for i in range(default_args.batch_chunk):
                data_i = data_chunks[i].contiguous()
                target_i = target_chunks[i].contiguous()
                ret = model(data_i, target_i, *mems[i])
                loss, mems[i] = ret[0], ret[1:]
                loss = loss.float().mean().type_as(loss) / default_args.batch_chunk
                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                train_losses.append(loss.float().item())
        else:
            ret = model(data, target, *mems)
            loss, mems = ret[0], ret[1:]
            loss = loss.float().mean().type_as(loss)
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            train_losses.append(loss.float().item())

        if args.fp16:
            torch.nn.utils.clip_grad_norm_(
                amp.master_params(optimizer), default_args.clip
            )
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), default_args.clip)

        optimizer.step()
        parent_model.compute += openai_compute(
            non_emb_param_count(parent_model, ntokens), data.numel(), 1
        )

        # step-wise learning rate annealing
        train_step += 1
        parent_model.training_steps += 1
        if default_args.scheduler in ["cosine", "constant", "dev_perf"]:
            # linear warmup stage
            if train_step < default_args.warmup_step:
                curr_lr = default_args.lr * train_step / default_args.warmup_step
                optimizer.param_groups = curr_lr
            else:
                if default_args.scheduler == "cosine":
                    scheduler.step(train_step)
        elif default_args.scheduler == "inv_sqrt":
            scheduler.step(train_step)

        if train_step == default_args.max_step:
            return (
                parent_model.compute * 24 * 3600,
                time.time() - start_time,
                train_step * data.numel(),
            )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="benchmarking script")

    parser.add_argument(
        "-l", "--n_layers", nargs="+", help="n_layers to test", required=True
    )
    parser.add_argument(
        "-d", "--d_models", nargs="+", help="d_models to test", required=True
    )
    parser.add_argument(
        "-b", "--batch_sizes", nargs="+", help="batch sizes to test", required=True
    )
    parser.add_argument("--fp16", type=str, default=None, choices=["O1", "O2", "O0"])
    parser.add_argument("-t", "--tracking", action="store_true")
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    start_time = time.time()
    if args.reload:
        try:
            results = json.load(open(f"compute_grid_results_{args.fp16}.json"))
            print(f"reloaded from compute_grid_results_{args.fp16}.json")
        except FileNotFoundError:
            results = {}
    else:
        results = {}

    default_args = DotDict(
        {
            "data": "../data/enwik8/",
            "dataset": "enwik8",
            "batch_chunk": 1,
            "not_tied": False,
            "div_val": 1,
            "pre_lnorm": False,
            "attn_type": 0,
            "dropout": 0.0,
            "dropatt": 0.0,
            "init": "normal",
            "emb_init": "normal",
            "init_range": 0.1,
            "emb_init_range": 0.01,
            "init_std": 0.02,
            "proj_init_std": 0.01,
            "optim": "adam",
            "lr": 5e-05,
            "mom": 0.0,
            "scheduler": "cosine",
            "warmup_step": 0,
            "decay_rate": 0.5,
            "lr_min": 0.0,
            "clip": 0.25,
            "clip_nonemb": False,
            "eta_min": 0.0,
            "tgt_len": 150,
            "eval_tgt_len": 150,
            "ext_len": 0,
            "mem_len": 150,
            "varlen": False,
            "same_length": False,
            "clamp_len": -1,
            "seed": 1111,
            "max_step": 100,
            "cuda": True,
            "multi_gpu": False,
            "gpu0_bsz": -1,
            "debug": False,
            "knockknock": True,
            "tied": True,
        }
    )

    device = torch.device("cuda" if default_args.cuda else "cpu")

    if args.fp16 == "O1":
        amp.register_half_function(torch, "einsum")

    cutoffs, tie_projs = [], [False]

    for n_layer, d_model, batch_size in product(
        args.n_layers, args.d_models, args.batch_sizes
    ):

        n_layer, d_model, batch_size = int(n_layer), int(d_model), int(batch_size)
        if args.reload:
            if results.get(str((n_layer, d_model, batch_size))) is not None:
                print(f"{(n_layer, d_model, batch_size)} already in results")
                continue

        corpus = get_lm_corpus(default_args.data, default_args.dataset)
        ntokens = len(corpus.vocab)
        default_args.n_token = ntokens

        if args.tracking:
            from experiment_impact_tracker.compute_tracker import ImpactTracker

            tracker = ImpactTracker(f"impact/{n_layer}_{d_model}_{batch_size}")
            tracker.launch_impact_monitor()

        n_head, d_head = head_repartition_rule(d_model)
        d_inner = d_model

        model = MemTransformerLM(
            ntokens,
            n_layer,
            n_head,
            d_model,
            d_head,
            d_inner,
            default_args.dropout,
            default_args.dropatt,
            tie_weight=default_args.tied,
            d_embed=d_model,
            div_val=default_args.div_val,
            tie_projs=tie_projs,
            pre_lnorm=default_args.pre_lnorm,
            tgt_len=default_args.tgt_len,
            ext_len=default_args.ext_len,
            mem_len=default_args.mem_len,
            cutoffs=cutoffs,
            same_length=default_args.same_length,
            attn_type=default_args.attn_type,
            clamp_len=default_args.clamp_len,
        )
        initialization_func = partial(
            weights_init,
            init="normal",
            init_range=0.1,
            init_std=0.02,
            proj_init_std=0.01,
        )
        model.apply(initialization_func)

        try:
            tr_iter = corpus.get_iterator(
                "train",
                batch_size,
                default_args.tgt_len,
                device=device,
                ext_len=default_args.ext_len,
            )
            para_model = parallelize_model(model, default_args)
            optimizers = build_optimizer(para_model, default_args, reload=False)
            optimizer, optimizer_sparse = optimizers
            schedulers = build_scheduler(optimizers, default_args)
            scheduler, scheduler_sparse = schedulers
            if default_args.cuda and args.fp16:
                para_model, optimizer = amp.initialize(
                    para_model, optimizer, opt_level=args.fp16, verbosity=0
                )

            compute, run_time, processed_tokens = benchmark(
                para_model, optimizers, schedulers
            )
            total_time = time.time() - start_time
            print("-" * 130)
            print(
                f"n_layer {n_layer} d_model {d_model} batch_size {batch_size} fp16 {args.fp16}: "
                + "{:.4e} FLOs in {:.4e}s for ".format(compute, run_time)
                + f"{processed_tokens} tokens, "
                f"total time {total_time}"
            )
            results[str((n_layer, d_model, batch_size))] = (
                compute,
                run_time,
                processed_tokens,
                compute / run_time,
            )

        except RuntimeError as e:
            print("-" * 100)
            total_time = time.time() - start_time
            print(
                f"n_layer {n_layer} d_model {d_model} batch_size {batch_size} fp16 {args.fp16}: OOM error, "
                f"total time {total_time}"
            )
            results[str((n_layer, d_model, batch_size))] = None

        finally:
            # Handle CUDA OOM Error Safely
            try:
                del model
                del para_model
                del optimizer
                del scheduler
                gc.collect()
                torch.cuda.empty_cache()
            except NameError:
                pass
        with open(f"compute_grid_results_{args.fp16}.json", "w") as f:
            json.dump(results, f, indent=2)

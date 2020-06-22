# coding: utf-8
import time
import math
import os

import itertools
from functools import partial
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from data_utils import get_time_series
from old.mem_transformer import MemTransformerLM

from utils.exp_utils import create_exp_dir
from utils.data_parallel import BalancedDataParallel
from utils.initialization import weights_init
from utils.argparsing import parser
from utils.torch_utils import non_emb_param_count, openai_compute


###############################################################################
##
## Helper functions
##
def update_dropout(m):
    classname = m.__class__.__name__
    if classname.find("Dropout") != -1:
        if hasattr(m, "p"):
            m.p = args.dropout
    if hasattr(m, "dropout_p"):
        m.dropout_p = args.dropatt


def update_dropatt(m):
    if hasattr(m, "dropatt"):
        m.dropatt.p = args.dropatt
    if hasattr(m, "dropatt_p"):
        m.dropatt_p = args.dropatt


def parallelize_model(model, args):
    # bit hacky, we re-instantiate device here to be able to import this
    # function elsewhere
    device = torch.device("cuda" if args.cuda else "cpu")
    if args.multi_gpu:
        model = model.to(device)
        if args.gpu0_bsz >= 0:
            para_model = BalancedDataParallel(
                args.gpu0_bsz // args.batch_chunk, model, dim=1
            ).to(device)
        else:
            para_model = nn.DataParallel(model, dim=1).to(device)
    else:
        para_model = model.to(device)

    return para_model


def train_ts(args):
    def build_scheduler(optimizers, args):
        optimizer, optimizer_sparse = optimizers
        scheduler_sparse = None

        if args.scheduler == "cosine":
            # here we do not set eta_min to lr_min to be backward compatible
            # because in previous versions eta_min is default to 0
            # rather than the default value of lr_min 1e-6
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, args.max_step, eta_min=args.eta_min
            )  # should use eta_min arg

        elif args.scheduler == "inv_sqrt":
            # originally used for Transformer (in Attention is all you need)
            def lr_lambda(step):
                # return a multiplier instead of a learning rate
                if step == 0 and args.warmup_step == 0:
                    return 1.0
                else:
                    return (
                        1.0 / (step ** 0.5)
                        if step > args.warmup_step
                        else step / (args.warmup_step ** 1.5)
                    )

            scheduler = optim.lr_scheduler.LambdaLR(optimizer,
                                                    lr_lambda=lr_lambda)

        elif args.scheduler == "dev_perf":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=args.decay_rate,
                patience=args.patience,
                min_lr=args.lr_min,
            )

        elif args.scheduler == "constant":
            pass

        else:
            raise ValueError(f"scheduler type {args.scheduler} not recognized")

        return scheduler, scheduler_sparse

    ###############################################################################
    # Training code
    ###############################################################################
    def evaluate(eval_iter, model):

        # Turn on evaluation mode which disables dropout.
        model.eval()

        # debug
        # If the model does not use memory at all, make the ext_len longer.
        # Otherwise, make the mem_len longer and keep the ext_len the same.
        # if default_args.mem_len == 0:
        #     model.reset_length(default_args.eval_tgt_len,
        #                        default_args.ext_len + default_args.tgt_len -
        #                        default_args.eval_tgt_len, default_args.mem_len)
        # else:
        #     model.reset_length(default_args.eval_tgt_len,
        #                        default_args.ext_len, default_args.mem_len +
        #                       default_args.tgt_len - default_args.eval_tgt_len)

        # Evaluation
        total_len, total_loss = 0, 0.0
        with torch.no_grad():
            mems = tuple()
            for i, (data, target, seq_len) in enumerate(eval_iter):
                if i >= args.max_eval_steps > 0:
                    break
                ret = model(data, target, *mems)
                loss, mems = ret[0], ret[1:]
                loss = loss.mean()
                total_loss += seq_len * loss.float().item()
                total_len += seq_len

        # Switch back to the training mode
        # model.reset_length(default_args.tgt_len, default_args.ext_len,
        # default_args.mem_len)
        model.train()

        return total_loss / total_len

    # reverse distillation util
    def get_original_batches(model, tr_iter, integration_length):
        model.eval()
        if args.batch_chunk > 1:
            mems = [None for _ in range(args.batch_chunk)]
            first_logits = [[] for _ in range(args.batch_chunk)]
        else:
            mems = None
            first_logits = []
        train_iter = tr_iter.get_varlen_iter() if args.varlen else tr_iter
        with torch.no_grad():
            for batch, (data, target, seq_len) in enumerate(train_iter):
                if batch == integration_length:
                    break
                if args.batch_chunk > 1:
                    data_chunks = torch.chunk(data, args.batch_chunk, 1)
                    for i in range(args.batch_chunk):
                        data_i = data_chunks[i].contiguous()
                        logits, mems[i] = model._forward(data_i, mems=mems[i])
                        first_logits[i].append(logits.cpu())
                else:
                    logits, mems = model._forward(data, mems=mems)
                    first_logits.append(logits.cpu())
        return first_logits

    def build_optimizer(model, args, reload=False):
        optimizer_sparse = None
        if args.optim.lower() == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=args.lr,
                                  momentum=args.mom)
        elif args.optim.lower() == "adam":
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
        elif args.optim.lower() == "adagrad":
            optimizer = optim.Adagrad(model.parameters(), lr=args.lr)
        else:
            raise ValueError(f"optimizer type {args.optim} not recognized")

        if reload:
            if args.restart_from is not None:
                optim_name = f"optimizer_{args.restart_from}.pt"
            else:
                optim_name = "optimizer.pt"
            optim_file_name = os.path.join(args.restart_dir, optim_name)
            logging(f"reloading {optim_file_name}")
            if os.path.exists(os.path.join(args.restart_dir, optim_name)):
                with open(
                        os.path.join(args.restart_dir, optim_name), "rb"
                ) as optim_file:
                    opt_state_dict = torch.load(optim_file)
                    try:
                        optimizer.load_state_dict(opt_state_dict)
                    # in case the optimizer param groups aren't the same shape,
                    # merge them
                    except:
                        logging("merging optimizer param groups")
                        opt_state_dict["param_groups"][0]["params"] = [
                            param
                            for param_group in opt_state_dict["param_groups"]
                            for param in param_group["params"]
                        ]
                        opt_state_dict["param_groups"] = [
                            opt_state_dict["param_groups"][0]
                        ]
                        optimizer.load_state_dict(opt_state_dict)
            else:
                logging("Optimizer was not saved. Start from scratch.")

        return optimizer, optimizer_sparse

    def log_val(val_loss, step, compute):
        logging("-" * 100)
        log_str = (
            "| Eval {:3d} at step {:>8d} | time: {:5.2f}s "
            "| valid loss {:5.2f}".format(
                step // args.eval_interval,
                step,
                (time.time() - eval_start_time),
                val_loss,
            )
        )
        log_str += " | bpc {:9.5f}".format(val_loss / math.log(2))
        logging(log_str)
        logging("-" * 100)

    def epoch_loop(
            epoch, model, optimizers, schedulers,
    ):
        nonlocal train_step

        # Turn on training mode which enables dropout.
        if isinstance(model, nn.DataParallel):
            parent_model = model.module
        else:
            parent_model = model
        optimizer, optimizer_sparse = optimizers
        scheduler, scheduler_sparse = schedulers

        # global train_step, best_val_loss, eval_start_time, log_start_time
        train_losses = []
        model.train()
        if args.batch_chunk > 1:
            mems = [tuple() for _ in range(args.batch_chunk)]
        else:
            mems = tuple()
        train_iter = tr_iter.get_varlen_iter() if args.varlen else tr_iter

        log_start_time = time.time()
        best_val_loss = float("Infinity")
        for batch, (data, target, seq_len) in enumerate(train_iter):
            model.zero_grad()
            if args.batch_chunk > 1:
                data_chunks = torch.chunk(data, args.batch_chunk, 1)
                target_chunks = torch.chunk(target, args.batch_chunk, 1)
                for i in range(args.batch_chunk):
                    data_i = data_chunks[i].contiguous()
                    target_i = target_chunks[i].contiguous()
                    ret = model(data_i, target_i, *mems[i])
                    loss, mems[i] = ret[0], ret[1:]
                    loss = loss.float().mean().type_as(loss) / args.batch_chunk
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
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer),
                                               args.clip)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

            optimizer.step()
            parent_model.compute += openai_compute(
                non_emb_param_count(parent_model, nseries), data.numel(), 1
            )

            # step-wise learning rate annealing
            train_step += 1
            parent_model.training_steps += 1

            # check for yet-to-thaw parameters
            if getattr(parent_model, "freeze_countdown", 0) > 0:
                parent_model.freeze_countdown -= 1

                # if this is the last step
                if parent_model.freeze_countdown == 0:
                    for parameter in parent_model.parameters():
                        parameter.requires_grad = True
                    logging("thawing all parameters")

            if args.scheduler in ["cosine", "constant", "dev_perf"]:
                # linear warmup stage
                if train_step < args.warmup_step:
                    curr_lr = args.lr * train_step / args.warmup_step
                    optimizer.param_groups = curr_lr
                else:
                    if args.scheduler == "cosine":
                        scheduler.step(train_step)
            elif args.scheduler == "inv_sqrt":
                scheduler.step(train_step)

            if train_step % args.log_interval == 0:
                cur_loss = np.mean(train_losses)
                elapsed = time.time() - log_start_time
                log_str = (
                    "| epoch {:3d} step {:>8d} "
                    "| {:>6d} batches "
                    "| lr {:.3g} "
                    "| ms/batch {:5.2f} "
                    "| loss {:5.2f}".format(
                        epoch,
                        train_step,
                        batch + 1,
                        optimizer.param_groups[0]["lr"],
                        elapsed * 1000 / args.log_interval,
                        cur_loss,
                    )
                )

                log_str += " | bpc {:9.5f}".format(cur_loss / math.log(2))
                logging(log_str)

                train_losses = []
                log_start_time = time.time()

            if train_step % args.eval_interval == 0:
                val_loss = evaluate(va_iter, model)
                log_val(val_loss, step=train_step, compute=parent_model.compute)
                # Save the model if the validation loss is the best we've seen so
                # far.
                if not best_val_loss or val_loss < best_val_loss:
                    best_val_loss = val_loss
                    if not args.debug:
                        if args.fp16:
                            with open(
                                    os.path.join(args.work_dir,
                                                 "amp_checkpoint.pt"), "wb",
                            ) as f:
                                checkpoint = {
                                    "model"    : model.state_dict(),
                                    "optimizer": optimizer.state_dict(),
                                    "amp"      : amp.state_dict(),
                                }
                                torch.save(checkpoint, f)
                        else:
                            with open(
                                    os.path.join(args.work_dir, "model.pt"),
                                    "wb"
                            ) as f:
                                torch.save(parent_model, f)
                            with open(
                                    os.path.join(args.work_dir, "optimizer.pt"),
                                    "wb",
                            ) as f:
                                torch.save(optimizer.state_dict(), f)

                # dev-performance based learning rate annealing
                if args.scheduler == "dev_perf":
                    scheduler.step(val_loss)

                eval_start_time = time.time()

            if train_step == args.max_step:
                break

    def expand_model(
            strategy,
            integration,
            integration_length,
            n_add,
            model: MemTransformerLM,
            optimizers,
            schedulers,
            tr_iter,
            va_iter,
            epoch,
            step,
    ):
        optimizer, _ = optimizers
        scheduler, _ = schedulers
        if integration:
            if not integration_length or integration_length <= 0:
                warnings.warn(
                    f"integration {integration} passed but integration_length is {integration_length}"
                )
            else:
                logging(
                    f"applying integration strategy {integration} with integration length {integration_length}"
                )

        # pre-expansion validation
        logging(f"evaluating before expanding")
        val_loss = evaluate(va_iter, model)
        log_val(val_loss, step=step, compute=model.compute)

        # infer example logits for reverse distillation
        if "reverse_distil" in integration:
            first_logits = get_original_batches(model, tr_iter,
                                                integration_length)

        # expansion
        logging(
            f"adding {n_add} layers before starting epoch {epoch} with method {strategy}"
        )
        new_layers = model.expand_layers(
            n_add, strategy=strategy, function=initialization_func
        )

        # optimizer update
        optimizer.add_param_group(
            {
                "params"    : new_layers.parameters(),
                "lr"        : optimizer.param_groups[0]["lr"],
                "initial_lr": optimizer.param_groups[0]["initial_lr"],
            }
        )
        scheduler.base_lrs.append(optimizer.param_groups[-1]["initial_lr"])

        # training loop for reverse distillation
        if "reverse_distil" in integration:
            fit_to_previous_model(model, new_layers, tr_iter, first_logits,
                                  integration)

        # freezing parameters for frozen restart, we do this afterwards else the
        # new layers get copied also without grads
        if "freeze" in integration and integration_length > 0:
            for param_group in optimizer.param_groups[:-1]:
                for parameter in param_group["params"]:
                    parameter.requires_grad = False
            model.freeze_countdown = integration_length

        # post-expansion validation
        logging(f"reevaluating")
        val_loss = evaluate(va_iter, model)
        log_val(val_loss, step=step, compute=model.compute)

    def expand_state(param, state):
        if param.shape != state.shape:
            ratios = [param.shape[i] // state.shape[i] for i in
                      range(len(param.shape))]
            return state.repeat(*ratios)
        else:
            return state

    def widen_model(
            strategy, ratio, model: MemTransformerLM, optimizers, va_iter,
            epoch, step,
    ):
        optimizer, _ = optimizers

        # pre-expansion validation
        logging(f"evaluating before widening")

        # debug
        val_loss = evaluate(va_iter, model)
        log_val(val_loss, compute=model.compute, step=step)

        # infer example logits for reverse distillation expansion
        logging(
            f"adding {ratio} layers before starting epoch {epoch} with method {strategy}"
        )
        model.add_heads(ratio, strategy=strategy, function=initialization_func)

        # optimizer update
        for param, states in optimizer.state.items():
            if isinstance(param, nn.Parameter):
                states["exp_avg"] = expand_state(param, states["exp_avg"])
                states["exp_avg_sq"] = expand_state(param, states["exp_avg_sq"])

        # training loop for reverse distillation
        # post-expansion validation
        logging(f"reevaluating")
        val_loss = evaluate(va_iter, model)
        log_val(val_loss, step=step, compute=model.compute)

    # reverse distillation trainer
    def fit_to_previous_model(model, new_layers, tr_iter, first_logits,
                              integration):
        mse_loss = torch.nn.MSELoss()
        if "partial" in integration:
            distil_optimizer, distil_optimizer_sparse = build_optimizer(
                new_layers, reload=False
            )
        else:
            distil_optimizer, distil_optimizer_sparse = build_optimizer(
                model, reload=False
            )
        if args.cuda and args.fp16:
            model, distil_optimizer = amp.initialize(
                model, distil_optimizer, opt_level=args.fp16
            )

        model.train()
        if args.batch_chunk > 1:
            mems = [None for _ in range(args.batch_chunk)]
        else:
            mems = None
        train_iter = tr_iter.get_varlen_iter() if args.varlen else tr_iter
        for batch, (data, _, _) in enumerate(train_iter):
            if batch == len(first_logits):
                break
            model.zero_grad()
            if args.batch_chunk > 1:
                data_chunks = torch.chunk(data, args.batch_chunk, 1)
                for i in range(args.batch_chunk):
                    data_i = data_chunks[i].contiguous()
                    logits, mems[i] = model._forward(data_i, mems=mems[i])
                    target_logits = first_logits[i][batch].to(logits.device)
                    loss = mse_loss(logits, target_logits) / args.batch_chunk
                    if args.fp16:
                        with amp.scale_loss(loss,
                                            distil_optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()
            else:
                logits, mems = model._forward(data, mems=mems)
                target_logits = first_logits[batch].to(logits.device)
                loss = mse_loss(logits, target_logits)
                if args.fp16:
                    with amp.scale_loss(loss, distil_optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

            if args.fp16:
                torch.nn.utils.clip_grad_norm_(
                    amp.master_params(distil_optimizer), args.clip
                )
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

            distil_optimizer.step()

    ###################################################################################
    #
    # main()
    #
    args.tied = not args.not_tied

    if args.d_embed < 0:
        args.d_embed = args.n_model

    # Validate `--fp16` option
    if args.fp16:
        if not args.cuda:
            print("WARNING: --fp16 requires --cuda, ignoring --fp16 option")
            args.fp16 = False
        else:
            try:
                from apex import amp

                if args.fp16 == "O1":
                    amp.register_half_function(torch, "einsum")
            except:
                print("WARNING: apex not installed, ignoring --fp16 option")
                args.fp16 = False

    device = torch.device("cuda" if args.cuda else "cpu")

    # Set the random seed manually for reproducibility.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print(
                "WARNING: You have a CUDA device, so you should probably run "
                "with --cuda "
            )
        else:
            torch.cuda.manual_seed_all(args.seed)

    ############################################################################
    # Logging
    ############################################################################

    assert args.ext_len >= 0, "extended context length must be non-negative"
    assert args.n_batch % args.batch_chunk == 0

    args.work_dir = "{}-{}".format(args.work_dir, args.dataset)
    args.work_dir = os.path.join(args.work_dir, time.strftime("%Y%m%d-%H%M%S"))
    logging = create_exp_dir(
        args.work_dir,
        scripts_to_save=["train_ts.py", "mem_transformer.py"],
        debug=args.debug,
    )

    ############################################################################
    # Load data
    ############################################################################
    time_series = get_time_series(args.datadir, args.dataset)
    nseries = len(time_series.vocab)
    args.n_token = nseries

    eval_batch_size = 20
    tr_iter = time_series.get_iterator(
        "train", args.n_batch, args.tgt_len, device=device,
        ext_len=args.ext_len,
    )
    va_iter = time_series.get_iterator(
        "valid",
        eval_batch_size,
        args.eval_tgt_len,
        device=device,
        ext_len=args.ext_len,
    )
    te_iter = time_series.get_iterator(
        "test", eval_batch_size, args.eval_tgt_len, device=device,
        ext_len=args.ext_len,
    )

    cutoffs, tie_projs = [], [False]

    ############################################################################
    # Define model
    ############################################################################

    initialization_func = partial(
        weights_init,
        init=args.init,
        init_range=args.init_range,
        init_std=args.init_std,
        proj_init_std=args.proj_init_std,
    )

    if args.restart and not args.fp16:
        if args.restart_from is not None:
            model_name = f"model_{args.restart_from}.pt"
        else:
            model_name = "model.pt"
        model_file_name = os.path.join(args.restart_dir, model_name)
        logging(f"reloading {model_file_name}")
        with open(model_file_name, "rb") as f:
            model = torch.load(f)
        # backwards compatibility with older saves
        if isinstance(model, nn.DataParallel):
            model = model.module
        model.backward_compatible(tie_weight=args.tied, tie_projs=tie_projs)
        if not args.fp16:
            model = model.float()
        model.apply(update_dropout)
        model.apply(update_dropatt)

    else:
        model = MemTransformerLM(
            nseries,
            args.n_layer,
            args.n_head,
            args.n_model,
            args.d_head,
            args.d_inner,
            args.dropout,
            args.dropatt,
            tie_weight=args.tied,
            d_embed=args.d_embed,
            div_val=args.div_val,
            tie_projs=tie_projs,
            pre_lnorm=args.pre_lnorm,
            tgt_len=args.tgt_len,
            ext_len=args.ext_len,
            mem_len=args.mem_len,
            cutoffs=cutoffs,
            same_length=args.same_length,
            clamp_len=args.clamp_len,
        )
        model.apply(initialization_func)

        # debug
        # model.word_emb.apply(initialization_func)
        # ensure embedding init is not overridden by out_layer in case of
        # weight sharing
    args.n_all_param = sum([p.nelement() for p in model.parameters()])
    args.n_nonemb_param = non_emb_param_count(model, nseries)

    logging("=" * 100)
    for k, v in args.__dict__.items():
        logging("    - {} : {}".format(k, v))
    logging("=" * 100)
    logging("#params = {}".format(args.n_all_param))
    logging("#non emb params = {}".format(args.n_nonemb_param))

    para_model = parallelize_model(model, args)
    optimizers = build_optimizer(
        para_model, args, reload=args.restart and not args.fp16
    )
    optimizer, optimizer_sparse = optimizers
    schedulers = build_scheduler(optimizers, args)
    scheduler, scheduler_sparse = schedulers

    if args.cuda and args.fp16:
        para_model, optimizer = amp.initialize(
            para_model, optimizer, opt_level=args.fp16
        )

        if args.restart:
            if args.restart_from is not None:
                checkpoint_name = f"amp_checkpoint_{args.restart_from}.pt"
            else:
                checkpoint_name = "amp_checkpoint.pt"
            with open(os.path.join(args.work_dir, checkpoint_name), "rb") as f:
                checkpoint = torch.load(f)
                model.load_state_dict(checkpoint["model"])
                optimizer.load_state_dict(checkpoint["optimizer"])
                amp.load_state_dict(checkpoint["amp"])

    ############################################################################
    # Training loop
    ############################################################################

    # Loop over epochs.
    if args.reset_lr:
        # then they're different and we use train_step only for the new lr
        # scheduling
        train_step = 0
        optimizer.defaults["lr"] = args.lr
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.lr
            param_group["initial_lr"] = args.lr
        scheduler.base_lrs = [args.lr] * len(scheduler.base_lrs)
    else:
        train_step = model.training_steps

    best_val_loss = None

    # Reload previous step number in case of default_args.restart
    if train_step > 0:
        logging(f"restarting from step {train_step}")

    log_start_time = time.time()
    eval_start_time = time.time()

    def run_training():
        nonlocal train_step

        for epoch in itertools.count(start=first_epoch):
            # we check before the training loop; expanding at epoch 0 means
            # before training (for debug purposes)
            if args.expand and str(epoch - 1) in args.expansion_dict:
                n_add = int(args.expansion_dict[str(epoch - 1)])
                expand_model(
                    args.expand,
                    args.integration,
                    args.integration_length,
                    n_add,
                    model,
                    optimizers,
                    schedulers,
                    tr_iter,
                    va_iter,
                    epoch,
                    train_step,
                )
            if args.widen and str(epoch - 1) in args.widen_dict:
                ratio = int(args.widen_dict[str(epoch - 1)])
                widen_model(
                    args.widen, ratio, model, optimizers, va_iter, epoch,
                    train_step,
                )
            epoch_loop(epoch, para_model, optimizers, schedulers)
            if train_step >= args.max_step:
                logging("-" * 100)
                logging("End of training")
                break
            if not args.debug and args.log_first_epochs:
                if epoch <= args.log_first_epochs:
                    logging(f"saving model at the end of epoch {epoch}")
                    if args.fp16:
                        with open(
                                os.path.join(args.work_dir,
                                             f"amp_checkpoint_{epoch}.pt"),
                                "wb",
                        ) as f:
                            checkpoint = {
                                "model"    : model.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "amp"      : amp.state_dict(),
                            }
                            torch.save(checkpoint, f)
                    else:
                        with open(
                                os.path.join(args.work_dir,
                                             f"model_{epoch}.pt"), "wb",
                        ) as f:
                            torch.save(model, f)
                        with open(
                                os.path.join(args.work_dir,
                                             f"optimizer_{epoch}.pt"), "wb",
                        ) as f:
                            torch.save(optimizer.state_dict(), f)

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        if args.restart_from:
            first_epoch = args.restart_from + 1
            print(f"restarting from epoch {first_epoch}")
        else:
            first_epoch = 1
        run_training()

    except KeyboardInterrupt:
        logging("-" * 100)
        logging("Exiting from training early")

    # Load the best model.
    if args.fp16:
        with open(os.path.join(args.work_dir, "amp_checkpoint.pt"), "rb") as f:
            checkpoint = torch.load(f)
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            amp.load_state_dict(checkpoint["amp"])
    else:
        with open(os.path.join(args.work_dir, "model.pt"), "rb") as f:
            model = torch.load(f)
    para_model = model.to(device)

    # Run on test data.
    test_loss = evaluate(te_iter, para_model)
    logging("=" * 100)
    logging(
        "| End of training | test loss {:5.2f} | test bpc {:9.5f}".format(
            test_loss, test_loss / math.log(2)
        )
    )
    logging("=" * 100)


if __name__ == "__main__":
    args = parser.parse_args()
    train_ts(args)

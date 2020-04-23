# coding: utf-8
import time
import math
import os, sys
import itertools
from functools import partial
import warnings

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from data_utils import get_lm_corpus
from mem_transformer import MemTransformerLM
from utils.exp_utils import create_exp_dir
from utils.data_parallel import BalancedDataParallel
from utils.initialization import weights_init
from utils.argparsing import parser

args = parser.parse_args()
args.tied = not args.not_tied

if args.d_embed < 0:
    args.d_embed = args.d_model

# Validate `--fp16` option
if args.fp16:
    if not args.cuda:
        print('WARNING: --fp16 requires --cuda, ignoring --fp16 option')
        args.fp16 = False
    else:
        try:
            from apex.fp16_utils import FP16_Optimizer
        except:
            print('WARNING: apex not installed, ignoring --fp16 option')
            args.fp16 = False

device = torch.device('cuda' if args.cuda else 'cpu')


###############################################################################
# Helper functions
###############################################################################


def update_dropout(m):
    classname = m.__class__.__name__
    if classname.find('Dropout') != -1:
        if hasattr(m, 'p'):
            m.p = args.dropout
    if hasattr(m, "dropout_p"):
        m.dropout_p = args.dropatt


def update_dropatt(m):
    if hasattr(m, 'dropatt'):
        m.dropatt.p = args.dropatt
    if hasattr(m, 'dropatt_p'):
        m.dropatt_p = args.dropatt


def parallelize_model(model):
    if args.multi_gpu:
        model = model.to(device)
        if args.gpu0_bsz >= 0:
            para_model = BalancedDataParallel(args.gpu0_bsz // args.batch_chunk,
                                              model, dim=1).to(device)
        else:
            para_model = nn.DataParallel(model, dim=1).to(device)
    else:
        para_model = model.to(device)

    return para_model


def build_optimizer(model):
    optimizer_sparse = None
    if args.optim.lower() == 'sgd':
        if args.sample_softmax > 0:
            dense_params, sparse_params = [], []
            for param in model.parameters():
                if param.size() == model.word_emb.weight.size():
                    sparse_params.append(param)
                else:
                    dense_params.append(param)
            optimizer_sparse = optim.SGD(sparse_params, lr=args.lr * 2)
            optimizer = optim.SGD(dense_params, lr=args.lr, momentum=args.mom)
        else:
            optimizer = optim.SGD(model.parameters(), lr=args.lr,
                                  momentum=args.mom)
    elif args.optim.lower() == 'adam':
        if args.sample_softmax > 0:
            dense_params, sparse_params = [], []
            for param in model.parameters():
                if param.size() == model.word_emb.weight.size():
                    sparse_params.append(param)
                else:
                    dense_params.append(param)
            optimizer_sparse = optim.SparseAdam(sparse_params, lr=args.lr)
            optimizer = optim.Adam(dense_params, lr=args.lr)
        else:
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optim.lower() == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=args.lr)
    else:
        raise ValueError(f"optimizer type {args.optim} not recognized")

    if args.restart:
        if args.restart_from is not None:
            optim_name = f'optimizer_{args.restart_from}.pt'
        else:
            optim_name = 'optimizer.pt'
        optim_file_name = os.path.join(args.restart_dir, optim_name)
        logging(f"reloading {optim_file_name}")
        if os.path.exists(os.path.join(args.restart_dir, optim_name)):
            with open(os.path.join(args.restart_dir, 'optimizer.pt'), 'rb') as optim_file:
                opt_state_dict = torch.load(optim_file)
                optimizer.load_state_dict(opt_state_dict)
        else:
            logging('Optimizer was not saved. Start from scratch.')

    return optimizer, optimizer_sparse


def build_scheduler(optimizers):
    optimizer, optimizer_sparse = optimizers
    scheduler_sparse = None
    if args.scheduler == 'cosine':
        # here we do not set eta_min to lr_min to be backward compatible
        # because in previous versions eta_min is default to 0
        # rather than the default value of lr_min 1e-6
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                         args.max_step, eta_min=args.eta_min)  # should use eta_min arg
        if args.sample_softmax > 0:
            scheduler_sparse = optim.lr_scheduler.CosineAnnealingLR(optimizer_sparse,
                                                                    args.max_step,
                                                                    eta_min=args.eta_min)  # should use eta_min arg
    elif args.scheduler == 'inv_sqrt':
        # originally used for Transformer (in Attention is all you need)
        def lr_lambda(step):
            # return a multiplier instead of a learning rate
            if step == 0 and args.warmup_step == 0:
                return 1.
            else:
                return 1. / (step ** 0.5) if step > args.warmup_step \
                    else step / (args.warmup_step ** 1.5)

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    elif args.scheduler == 'dev_perf':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         factor=args.decay_rate, patience=args.patience,
                                                         min_lr=args.lr_min)
        if args.sample_softmax > 0:
            scheduler_sparse = optim.lr_scheduler.ReduceLROnPlateau(optimizer_sparse,
                                                                    factor=args.decay_rate, patience=args.patience,
                                                                    min_lr=args.lr_min)
    elif args.scheduler == 'constant':
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
    # if args.mem_len == 0:
    #     model.reset_length(args.eval_tgt_len,
    #                        args.ext_len + args.tgt_len - args.eval_tgt_len, args.mem_len)
    # else:
    #     model.reset_length(args.eval_tgt_len,
    #                        args.ext_len, args.mem_len + args.tgt_len - args.eval_tgt_len)

    # Evaluation
    total_len, total_loss = 0, 0.
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
    # model.reset_length(args.tgt_len, args.ext_len, args.mem_len)
    model.train()

    return total_loss / total_len


def log_val(val_loss):
    logging('-' * 100)
    log_str = '| Eval {:3d} at step {:>8d} | time: {:5.2f}s ' \
              '| valid loss {:5.2f}'.format(
        train_step // args.eval_interval, train_step,
        (time.time() - eval_start_time), val_loss)
    if args.dataset in ['enwik8', 'text8']:
        log_str += ' | bpc {:9.5f}'.format(val_loss / math.log(2))
    else:
        log_str += ' | valid ppl {:9.3f}'.format(math.exp(val_loss))
    logging(log_str)
    if args.wandb:
        wandb.log({"valid_loss": val_loss}, step=train_step)
    logging('-' * 100)


def train(model, optimizers, schedulers):
    # Turn on training mode which enables dropout.
    optimizer, optimizer_sparse = optimizers
    scheduler, scheduler_sparse = schedulers
    global train_step, train_loss, best_val_loss, eval_start_time, log_start_time
    model.train()
    if args.batch_chunk > 1:
        mems = [tuple() for _ in range(args.batch_chunk)]
    else:
        mems = tuple()
    train_iter = tr_iter.get_varlen_iter() if args.varlen else tr_iter
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
                    optimizer.backward(loss)
                else:
                    loss.backward()
                train_loss += loss.float().item()
        else:
            ret = model(data, target, *mems)
            loss, mems = ret[0], ret[1:]
            loss = loss.float().mean().type_as(loss)
            if args.fp16:
                optimizer.backward(loss)
            else:
                loss.backward()
            train_loss += loss.float().item()

        if args.fp16:
            optimizer.clip_master_grads(args.clip)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        optimizer.step()
        if args.sample_softmax > 0:
            optimizer_sparse.step()

        # step-wise learning rate annealing
        train_step += 1
        try:
            model.training_steps += 1
        # DataParallel
        except AttributeError:
            model.module.training_steps += 1
        if args.scheduler in ['cosine', 'constant', 'dev_perf']:
            # linear warmup stage
            if train_step < args.warmup_step:
                curr_lr = args.lr * train_step / args.warmup_step
                optimizer.param_groups = curr_lr
                if args.sample_softmax > 0:
                    optimizer_sparse.param_groups[0]['lr'] = curr_lr * 2
            else:
                if args.scheduler == 'cosine':
                    scheduler.step(train_step)
                    if args.sample_softmax > 0:
                        scheduler_sparse.step(train_step)
        elif args.scheduler == 'inv_sqrt':
            scheduler.step(train_step)

        if train_step % args.log_interval == 0:
            cur_loss = train_loss / args.log_interval
            elapsed = time.time() - log_start_time
            log_str = '| epoch {:3d} step {:>8d} | {:>6d} batches | lr {:.3g} ' \
                      '| ms/batch {:5.2f} | loss {:5.2f}'.format(
                epoch, train_step, batch + 1, optimizer.param_groups[0]['lr'],
                                   elapsed * 1000 / args.log_interval, cur_loss)
            if args.dataset in ['enwik8', 'text8']:
                log_str += ' | bpc {:9.5f}'.format(cur_loss / math.log(2))
            else:
                log_str += ' | ppl {:9.3f}'.format(math.exp(cur_loss))
            logging(log_str)
            if args.wandb:
                wandb.log({"train_loss": cur_loss, "learning rate": optimizer.param_groups[0]['lr']}, step=train_step)
            train_loss = 0
            log_start_time = time.time()

        if train_step % args.eval_interval == 0:
            val_loss = evaluate(va_iter, model)
            log_val(val_loss)
            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                if not args.debug:
                    with open(os.path.join(args.work_dir, 'model.pt'), 'wb') as f:
                        torch.save(model, f)
                    with open(os.path.join(args.work_dir, 'optimizer.pt'), 'wb') as f:
                        torch.save(optimizer.state_dict(), f)
                best_val_loss = val_loss

            # dev-performance based learning rate annealing
            if args.scheduler == 'dev_perf':
                scheduler.step(val_loss)
                if args.sample_softmax > 0:
                    scheduler_sparse.step(val_loss)

            eval_start_time = time.time()

        if train_step == args.max_step:
            break


def expand_model(strategy, integration, integration_length, n_add, model, optimizers, schedulers, tr_iter, va_iter):
    optimizer, _ = optimizers
    scheduler, _ = schedulers
    if integration and (not integration_length or integration_length <= 0):
        warnings.warn(f"integration {integration} passed but integration_length is {integration_length}")
    # pre-expansion validation
    logging(f"evaluating before expanding")
    val_loss = evaluate(va_iter, model)
    log_val(val_loss)
    # infer example logits for reverse distillation
    if "reverse_distil" in integration:
        first_logits = get_original_batches(model, tr_iter, integration_length)
    # expansion
    logging(f"adding {n_add} layers before starting epoch {epoch} with method {strategy}")
    new_layers = model.expand_layers(n_add, strategy=strategy, function=initialization_func)
    # optimizer update
    optimizer.add_param_group({'params': new_layers.parameters(), 'lr': optimizer.param_groups[0]["lr"],
                               'initial_lr': optimizer.param_groups[0]["initial_lr"]})
    scheduler.base_lrs.append(optimizer.param_groups[-1]["initial_lr"])
    # training loop for reverse distillation
    if "reverse_distil" in integration:
        fit_to_previous_model(model, new_layers, tr_iter, first_logits, integration)
    # post-expansion validation
    logging(f"reevaluating")
    val_loss = evaluate(va_iter, model)
    log_val(val_loss)


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
                    first_logits[i].append(logits)
            else:
                logits, mems = model._forward(data, mems=mems)
                first_logits.append(logits)
    return first_logits


# reverse distillation trainer
def fit_to_previous_model(model, new_layers, tr_iter, first_logits, integration):
    mse_loss = torch.nn.MSELoss()
    if "partial" in integration:
        distil_optimizer, distil_optimizer_sparse = build_optimizer(new_layers)
    else:
        distil_optimizer, distil_optimizer_sparse = build_optimizer(model)
    if args.cuda and args.fp16:
        distil_optimizer = FP16_Optimizer(distil_optimizer,
                                   static_loss_scale=args.static_loss_scale,
                                   dynamic_loss_scale=args.dynamic_loss_scale,
                                   dynamic_loss_args={'init_scale': 2 ** 16})
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
                target_logits = first_logits[i][batch]
                loss = mse_loss(logits, target_logits) / args.batch_chunk
                if args.fp16:
                    distil_optimizer.backward(loss)
                else:
                    loss.backward()
        else:
            logits, mems = model._forward(data, mems=mems)
            target_logits = first_logits[batch]
            loss = mse_loss(logits, target_logits)
            if args.fp16:
                distil_optimizer.backward(loss)
            else:
                loss.backward()

        if args.fp16:
            distil_optimizer.clip_master_grads(args.clip)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        distil_optimizer.step()
        if args.sample_softmax > 0:
            distil_optimizer_sparse.step()


if __name__ == "__main__":

    # Set the random seed manually for reproducibility.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print('WARNING: You have a CUDA device, so you should probably run with --cuda')
        else:
            torch.cuda.manual_seed_all(args.seed)

    ###############################################################################
    # Logging
    ###############################################################################

    assert args.ext_len >= 0, 'extended context length must be non-negative'
    assert args.batch_size % args.batch_chunk == 0

    args.work_dir = '{}-{}'.format(args.work_dir, args.dataset)
    args.work_dir = os.path.join(args.work_dir, time.strftime('%Y%m%d-%H%M%S'))
    logging = create_exp_dir(args.work_dir,
                             scripts_to_save=['train.py', 'mem_transformer.py'], debug=args.debug)

    if args.wandb:
        import wandb

        logged_params = {"dataset": args.dataset,
                         "sequence_length": args.tgt_len,
                         "memory_length": args.mem_len,
                         "n_embd": args.d_model,
                         "d_inner": args.d_inner,
                         "n_layer": args.n_layer,
                         "n_head": args.n_head,
                         "dropout": args.dropout,
                         "div_val": args.div_val,
                         "codebase": "CMU",
                         "expansion_style": args.expand,
                         "expansion_times": args.expansion_dict
                         }
        wandb.init(project="salamander", config=logged_params)

    ###############################################################################
    # Load data
    ###############################################################################

    corpus = get_lm_corpus(args.data, args.dataset)
    ntokens = len(corpus.vocab)
    args.n_token = ntokens

    eval_batch_size = 10
    tr_iter = corpus.get_iterator('train', args.batch_size, args.tgt_len,
                                  device=device, ext_len=args.ext_len)
    va_iter = corpus.get_iterator('valid', eval_batch_size, args.eval_tgt_len,
                                  device=device, ext_len=args.ext_len)
    te_iter = corpus.get_iterator('test', eval_batch_size, args.eval_tgt_len,
                                  device=device, ext_len=args.ext_len)

    # adaptive softmax / embedding
    cutoffs, tie_projs = [], [False]
    if args.adaptive:
        assert args.dataset in ['wt103', 'lm1b']
        if args.dataset == 'wt103':
            cutoffs = [20000, 40000, 200000]
            tie_projs += [True] * len(cutoffs)
        elif args.dataset == 'lm1b':
            cutoffs = [60000, 100000, 640000]
            tie_projs += [False] * len(cutoffs)

    ###############################################################################
    # Define model
    ###############################################################################

    initialization_func = partial(weights_init,
                                  init=args.init,
                                  init_range=args.init_range,
                                  init_std=args.init_std,
                                  proj_init_std=args.proj_init_std)
    if args.restart:
        if args.restart_from is not None:
            model_name = f'model_{args.restart_from}.pt'
        else:
            model_name = 'model.pt'
        model_file_name = os.path.join(args.restart_dir, model_name)
        logging(f"reloading {model_file_name}")
        with open(model_file_name, 'rb') as f:
            model = torch.load(f)
        if not args.fp16:
            model = model.float()
        model.apply(update_dropout)
        model.apply(update_dropatt)
    else:
        model = MemTransformerLM(ntokens, args.n_layer, args.n_head, args.d_model,
                                 args.d_head, args.d_inner, args.dropout, args.dropatt,
                                 tie_weight=args.tied, d_embed=args.d_embed, div_val=args.div_val,
                                 tie_projs=tie_projs, pre_lnorm=args.pre_lnorm, tgt_len=args.tgt_len,
                                 ext_len=args.ext_len, mem_len=args.mem_len, cutoffs=cutoffs,
                                 same_length=args.same_length, attn_type=args.attn_type,
                                 clamp_len=args.clamp_len, sample_softmax=args.sample_softmax)
        model.apply(initialization_func)
        # debug
        # model.word_emb.apply(initialization_func)
        # ensure embedding init is not overridden by out_layer in case of weight sharing
    args.n_all_param = sum([p.nelement() for p in model.parameters()])
    args.n_nonemb_param = sum([p.nelement() for p in model.layers.parameters()])

    logging('=' * 100)
    for k, v in args.__dict__.items():
        logging('    - {} : {}'.format(k, v))
    logging('=' * 100)
    logging('#params = {}'.format(args.n_all_param))
    logging('#non emb params = {}'.format(args.n_nonemb_param))

    if args.fp16:
        model = model.half()

    para_model = parallelize_model(model)
    optimizers = build_optimizer(para_model)
    optimizer, optimizer_sparse = optimizers
    schedulers = build_scheduler(optimizers)
    scheduler, scheduler_sparse = schedulers

    if args.cuda and args.fp16:
        # If args.dynamic_loss_scale is False, static_loss_scale will be used.
        # If args.dynamic_loss_scale is True, it will take precedence over static_loss_scale.
        optimizer = FP16_Optimizer(optimizer,
                                   static_loss_scale=args.static_loss_scale,
                                   dynamic_loss_scale=args.dynamic_loss_scale,
                                   dynamic_loss_args={'init_scale': 2 ** 16})

    ###############################################################################
    # Training loop
    ###############################################################################

    # Loop over epochs.
    train_step = model.training_steps
    train_loss = 0
    best_val_loss = None
    # Reload previous step number in case of args.restart
    if train_step > 0:
        logging(f"restarting from step {train_step}")

    log_start_time = time.time()
    eval_start_time = time.time()

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        if args.restart_from:
            first_epoch = args.restart_from + 1
            print(f"restarting from epoch {first_epoch}")
        else:
            first_epoch = 1
        for epoch in itertools.count(start=first_epoch):
            # we check before the training loop; expanding at epoch 0 means before training (for debug purposes)
            if args.expand and str(epoch - 1) in args.expansion_dict:
                n_add = int(args.expansion_dict[str(epoch - 1)])
                expand_model(args.expand, args.integration, args.integration_length,
                             n_add, model, optimizers, schedulers, tr_iter, va_iter)
            train(para_model, optimizers, schedulers)
            if train_step >= args.max_step:
                logging('-' * 100)
                logging('End of training')
                break
            if not args.debug and args.log_first_epochs:
                if epoch <= args.log_first_epochs:
                    logging(f"saving model at the end of epoch {epoch}")
                    with open(os.path.join(args.work_dir, f'model_{epoch}.pt'), 'wb') as f:
                        torch.save(model, f)
                    with open(os.path.join(args.work_dir, f'optimizer_{epoch}.pt'), 'wb') as f:
                        torch.save(optimizer.state_dict(), f)


    except KeyboardInterrupt:
        logging('-' * 100)
        logging('Exiting from training early')

    # Load the bestzdel.
    with open(os.path.join(args.work_dir, 'model.pt'), 'rb') as f:
        model = torch.load(f)
    para_model = model.to(device)

    # Run on test data.
    test_loss = evaluate(te_iter, para_model)
    logging('=' * 100)
    if args.dataset in ['enwik8', 'text8']:
        logging('| End of training | test loss {:5.2f} | test bpc {:9.5f}'.format(
            test_loss, test_loss / math.log(2)))
    else:
        logging('| End of training | test loss {:5.2f} | test ppl {:9.3f}'.format(
            test_loss, math.exp(test_loss)))
    wandb.log({"test_loss": test_loss})
    logging('=' * 100)

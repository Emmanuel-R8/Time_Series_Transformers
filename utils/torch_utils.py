import torch


def repeat_parameter(parameter, *sizes):
    parameter.data = parameter.data.repeat(*sizes)
    if parameter.grad is not None:
        parameter.grad = parameter.grad.repeat(*sizes)


def qkv_weight_repeat(parameter, ratio):
    q, k, v = torch.chunk(parameter, 3, dim=0)
    q = q.repeat(ratio, ratio)
    k = k.repeat(ratio, ratio)
    v = v.repeat(ratio, ratio)
    return torch.cat([q, k, v], dim=0)


def expand_qkv(qkv_net, ratio):
    qkv_net.weight.data = qkv_weight_repeat(qkv_net.weight.data, ratio)
    if qkv_net.weight.grad is not None:
        qkv_net.weight.grad = qkv_weight_repeat(qkv_net.weight.grad, ratio)


def openai_compute(n_params, batch_size, training_steps):
    # given in PF/s (hence the / 24 / 3600)
    return 6 * n_params * batch_size * training_steps / 24 / 3600


def excluded_from_params(parameter: torch.nn.Parameter, vocab_size=-1):
    return vocab_size in parameter.shape


def non_emb_param_count(model: torch.nn.Module, vocab_size=-1):
    return sum(
        p.numel() for p in model.parameters() if not excluded_from_params(p, vocab_size)
    )

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

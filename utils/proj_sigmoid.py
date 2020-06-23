import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_lightning.core.lightning import LightningModule

if torch.cuda.is_available():
    CUDA_MAJOR = int(torch.version.cuda.split(".")[0])
    CUDA_MINOR = int(torch.version.cuda.split(".")[1])


class ProjectedSigmoid(LightningModule):
    def __init__(self, n_series, d_model, d_proj, keep_order=False):
        super(ProjectedSigmoid, self).__init__()

        self.n_series = n_series
        self.d_model = d_model
        self.d_proj = d_proj

        self.out_layers = nn.ModuleList()
        self.out_projs = nn.ParameterList()

        if d_proj != d_model:
            self.out_projs.append(nn.Parameter(torch.Tensor(d_proj, d_model)))

        self.out_layers.append(nn.Linear(d_model, n_series))

    def _compute_logit(self, hidden, weight, bias, proj):
        if proj is None:
            logit = F.linear(hidden, weight, bias=bias)
        else:
            # if CUDA_MAJOR <= 9 and CUDA_MINOR <= 1:
            proj_hid = F.linear(hidden, proj.t().contiguous())
            logit = F.linear(proj_hid, weight, bias=bias)
            # else:
            #     logit = torch.einsum('bd,de,ev->bv', (hidden, proj, weight.t()))
            #     if bias is not None:
            #         logit = logit + bias

        return logit

    def forward(self, hidden, target):
        """
            hidden :: [len*n_batch x d_proj]
            target :: [len*n_batch]
        """

        if hidden.size(0) != target.size(0):
            raise RuntimeError(
                "Input and target should have the same size " "in the batch dimension."
            )

        logit = self._compute_logit(
            hidden,
            self.out_layers[0].weight,
            self.out_layers[0].bias,
            self.out_projs[0],
        )

        nll = -F.logsigmoid(logit).gather(1, target.unsqueeze(1)).squeeze(1)

        return nll

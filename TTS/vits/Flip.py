from .utils import *
import torch

class Flip(nn.Module):
    def __init__(self, v_DEVICE):
        super().__init__()
        self.register_buffer('zero_logdet', torch.tensor(0.0, device=v_DEVICE))

    def forward(self, x, *args, reverse=False, **kwargs):
        x = torch.flip(x, [1])
        if not reverse:
            logdet = self.zero_logdet.expand(x.size(0))
            return x, logdet
        else:
            return x

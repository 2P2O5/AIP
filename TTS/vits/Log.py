from .utils import *

class Log(nn.Module):
    def __init__(self, v_DEVICE):
        super().__init__()

    def forward(self, x, x_mask, reverse=False, **kwargs):
        if not reverse:
            y = torch.log1p(torch.clamp_min(x - 1, 0)) * x_mask
            logdet = torch.sum(-y * x_mask, dim=(1, 2))
            return y, logdet
        else:
            x = torch.expm1(x) * x_mask + x_mask
            return x

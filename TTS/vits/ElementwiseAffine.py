from .utils import *

class ElementwiseAffine(nn.Module):
    def __init__(self, v_DEVICE, channels):
        super().__init__()
        self.channels = channels
        self.v_DEVICE = v_DEVICE
        self.m = nn.Parameter(torch.zeros(channels, 1, device=v_DEVICE))
        self.logs = nn.Parameter(torch.zeros(channels, 1, device=v_DEVICE))

    def forward(self, x, x_mask, reverse=False, **kwargs):
        m, logs = self.m, self.logs
        exp_logs = torch.exp(logs)

        if not reverse:
            y = m + exp_logs * x
            y *= x_mask
            logdet = torch.sum(logs * x_mask, dim=(1, 2))
            return y, logdet
        else:
            x = (x - m) * torch.exp(-logs) * x_mask
            return x

from .utils import *
from .Encoder import Encoder

class TransformerCouplingLayer(nn.Module):
    def __init__(
        self,
        v_DEVICE,
        channels,
        hidden_channels,
        kernel_size,
        n_layers,
        n_heads,
        p_dropout=0,
        filter_channels=0,
        mean_only=False,
        wn_sharing_parameter=None,
        gin_channels=0,
    ):
        assert channels % 2 == 0, "channels should be divisible by 2"
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.half_channels = channels // 2
        self.mean_only = mean_only

        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = (
            Encoder(
                v_DEVICE,
                hidden_channels,
                filter_channels,
                n_heads,
                n_layers,
                kernel_size,
                p_dropout,
                isflow=True,
                gin_channels=gin_channels,
            )
            if wn_sharing_parameter is None
            else wn_sharing_parameter
        )
        self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)
        nn.init.zeros_(self.post.weight)
        nn.init.zeros_(self.post.bias)

    def forward(self, x, x_mask, g=None, reverse=False):
        x0, x1 = torch.split(x, self.half_channels, dim=1)
        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask, g=g)
        stats = self.post(h) * x_mask

        if not self.mean_only:
            m, logs = torch.split(stats, self.half_channels, dim=1)
        else:
            m = stats
            logs = torch.zeros_like(m, device=x.device, dtype=x.dtype)

        if not reverse:
            x1 = (m + x1 * torch.exp(logs)) * x_mask
            logdet = torch.sum(logs * x_mask, dim=[1, 2])
        else:
            x1 = ((x1 - m) * torch.exp(-logs)) * x_mask
            logdet = None

        x = torch.cat([x0, x1], dim=1)
        return (x, logdet) if logdet is not None else x

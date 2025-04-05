from .utils import *
from .Flip import Flip
from .TransformerCouplingLayer import TransformerCouplingLayer

class TransformerCouplingBlock(nn.Module):
    def __init__(
        self,
        v_DEVICE,
        channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        n_flows=4,
        gin_channels=0,
        share_parameter=False,
    ):
        super().__init__()
        self.v_DEVICE = v_DEVICE
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()

        if share_parameter:
            raise RuntimeError

        self.wn = None

        for _ in range(n_flows):
            self.flows.extend([
                TransformerCouplingLayer(
                    v_DEVICE,
                    channels,
                    hidden_channels,
                    kernel_size,
                    n_layers,
                    n_heads,
                    p_dropout,
                    filter_channels,
                    mean_only=True,
                    wn_sharing_parameter=self.wn,
                    gin_channels=self.gin_channels,
                ),
                Flip(v_DEVICE)
            ])

    def forward(self, x, x_mask, g=None, reverse=False):
        flows = reversed(self.flows) if reverse else self.flows
        for flow in flows:
            x = flow(x, x_mask, g=g, reverse=reverse) if reverse else flow(x, x_mask, g=g, reverse=reverse)[0]
        return x

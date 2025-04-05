from .utils import *
from .DDSConv import DDSConv
from .Log import Log
from .ConvFlow import ConvFlow
from .ElementwiseAffine import ElementwiseAffine
from .Flip import Flip

class StochasticDurationPredictor(nn.Module):
    def __init__(
        self,
        v_DEVICE,
        in_channels,
        filter_channels,
        kernel_size,
        p_dropout,
        n_flows=4,
        gin_channels=0,
    ):
        super().__init__()
        
        self.v_DEVICE = v_DEVICE
        filter_channels = in_channels  
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.log_flow = Log(v_DEVICE)
        self.flows = nn.ModuleList()
        self.flows.append(ElementwiseAffine(v_DEVICE, 2))
        for _ in range(n_flows):
            self.flows.append(
                ConvFlow(v_DEVICE, 2, filter_channels, kernel_size, n_layers=3)
            )
            self.flows.append(Flip(v_DEVICE))

        self.post_pre = nn.Conv1d(1, filter_channels, 1)
        self.post_proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.post_convs = DDSConv(
            v_DEVICE, filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout
        )
        self.post_flows = nn.ModuleList()
        self.post_flows.append(ElementwiseAffine(v_DEVICE, 2))
        for i in range(4):
            self.post_flows.append(
                ConvFlow(v_DEVICE, 2, filter_channels, kernel_size, n_layers=3)
            )
            self.post_flows.append(Flip(v_DEVICE))

        self.pre = nn.Conv1d(in_channels, filter_channels, 1)
        self.proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.convs = DDSConv(
            v_DEVICE, filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout
        )
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, filter_channels, 1)

        self.log_2pi = math.log(2 * math.pi)

    def forward(self, x, x_mask, w=None, g=None, reverse=False, noise_scale=1.0):
        x = torch.detach(x)
        x = self.pre(x)
        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g)
        x = self.convs(x, x_mask)
        x = self.proj(x) * x_mask

        if not reverse:
            flows = self.flows
            assert w is not None

            logdet_tot_q = 0
            h_w = self.post_proj(self.post_convs(self.post_pre(w), x_mask)) * x_mask
            e_q = torch.randn_like(w).to(device=self.v_DEVICE) * x_mask
            z_q = e_q

            for flow in self.post_flows:
                z_q, logdet_q = flow(z_q, x_mask, g=(x + h_w))
                logdet_tot_q += logdet_q

            z_u, z1 = torch.split(z_q, [1, 1], 1)
            u = torch.sigmoid(z_u) * x_mask
            z0 = (w - u) * x_mask
            logdet_tot_q += torch.sum(
                (F.logsigmoid(z_u) + F.logsigmoid(-z_u)) * x_mask, dim=[1, 2]
            )
            logq = (
                torch.sum(-0.5 * (self.log_2pi + e_q**2) * x_mask, dim=[1, 2])
                - logdet_tot_q
            )

            logdet_tot = 0
            z0, logdet = self.log_flow(z0, x_mask)
            logdet_tot += logdet
            z = torch.cat([z0, z1], dim=1)

            for flow in flows:
                z, logdet = flow(z, x_mask, g=x, reverse=reverse)
                logdet_tot += logdet

            nll = (
                torch.sum(0.5 * (self.log_2pi + z**2) * x_mask, dim=[1, 2])
                - logdet_tot
            )
            return nll + logq
        else:
            flows = self.flows[:-2] + [self.flows[-1]]
            flows = reversed(flows)
            z = (
                torch.randn(x.size(0), 2, x.size(2), device=self.v_DEVICE, dtype=x.dtype)
                * noise_scale
            )
            for flow in flows:
                z = flow(z, x_mask, g=x, reverse=reverse)
            z0, z1 = torch.split(z, [1, 1], dim=1)
            logw = z0
            return logw

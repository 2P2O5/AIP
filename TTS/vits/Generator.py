from .utils import *

from .ResBlock1 import ResBlock1
from .ResBlock2 import ResBlock2
class Generator(torch.nn.Module):
    def __init__(
        self,
        v_DEVICE,
        initial_channel,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        gin_channels=0,
    ):
        super(Generator, self).__init__()
        self.v_DEVICE = v_DEVICE
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = Conv1d(
            initial_channel, upsample_initial_channel, 7, 1, padding=3
        )
        resblock_cls = ResBlock1 if resblock == "1" else ResBlock2

        self.ups = nn.ModuleList([
            weight_norm(
                ConvTranspose1d(
                    upsample_initial_channel // (2**i),
                    upsample_initial_channel // (2 ** (i + 1)),
                    k,
                    u,
                    padding=(k - u) // 2,
                )
            )
            for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes))
        ])

        self.resblocks = nn.ModuleList([
            resblock_cls(v_DEVICE, upsample_initial_channel // (2 ** (i + 1)), k, d)
            for i in range(len(self.ups))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes)
        ])

        self.conv_post = Conv1d(
            upsample_initial_channel // (2 ** len(upsample_rates)), 1, 7, 1, padding=3, bias=False
        )
        self.ups.apply(init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, g=None):
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            resblock_outputs = [
                self.resblocks[i * self.num_kernels + j](x)
                for j in range(self.num_kernels)
            ]
            x = sum(resblock_outputs) / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print("Removing weight norm...")
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()

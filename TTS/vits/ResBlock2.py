from .utils import *

class ResBlock2(torch.nn.Module):
    def __init__(self, v_DEVICE, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.convs = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=d,
                        padding=get_padding(kernel_size, d),
                    )
                )
                for d in dilation
            ]
        )
        self.convs.apply(init_weights)

    def forward(self, x, x_mask=None):
        for c in self.convs:
            x = F.leaky_relu(x, LRELU_SLOPE)
            if x_mask is not None:
                x *= x_mask
            x = c(x) + x
        if x_mask is not None:
            x *= x_mask
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)

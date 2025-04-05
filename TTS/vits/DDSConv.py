from .utils import *
from .LayerNorm import LayerNorm
import torch.jit

class DDSConv(nn.Module):
    """
    Dialted and Depth-Separable Convolution
    """
    def __init__(self, v_DEVICE, channels, kernel_size, n_layers, p_dropout=0.0):
        super().__init__()
        self.v_DEVICE = v_DEVICE
        self.channels = channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.p_dropout = p_dropout

        self.drop = nn.Dropout(p_dropout)
        self.convs_sep = nn.ModuleList()
        self.convs_1x1 = nn.ModuleList()
        self.norms_1 = nn.ModuleList()
        self.norms_2 = nn.ModuleList()
        self.activation = nn.GELU()
        for i in range(n_layers):
            dilation = kernel_size**i
            padding = (kernel_size * dilation - dilation) // 2
            self.convs_sep.append(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    groups=channels,
                    dilation=dilation,
                    padding=padding
                )
            )
            self.convs_1x1.append(nn.Conv1d(channels, channels, 1))
            self.norms_1.append(LayerNorm(v_DEVICE, channels))
            self.norms_2.append(LayerNorm(v_DEVICE, channels))

    def forward(self, x, x_mask, g=None):
        if g is not None:
            x = x + g
        for i in range(self.n_layers):
            y = self.convs_sep[i](x * x_mask)
            y = self.norms_1[i](y)
            y = self.activation(y)
            y = self.convs_1x1[i](y)
            y = self.norms_2[i](y)
            y = self.activation(y)
            y = self.drop(y)
            x = x + y
        return x * x_mask
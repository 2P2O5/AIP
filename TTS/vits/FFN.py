from .utils import *

class FFN(nn.Module):
    def __init__(
        self,
        v_DEVICE,
        in_channels,
        out_channels,
        filter_channels,
        kernel_size,
        p_dropout=0.0,
        activation=None,
        causal=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.activation = activation
        self.causal = causal

        self.padding_fn = self._causal_padding if causal else self._same_padding

        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=0)
        self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size, padding=0)
        self.drop = nn.Dropout(p_dropout)

        if activation == "gelu":
            self.activation_fn = lambda x: x * torch.sigmoid(1.702 * x)
        else:
            self.activation_fn = torch.relu

    def forward(self, x, x_mask):
        x_masked = x * x_mask
        x = self.conv_1(self.padding_fn(x_masked))
        x = self.activation_fn(x)
        x = self.drop(x)
        x = self.conv_2(self.padding_fn(x * x_mask))
        return x * x_mask

    def _causal_padding(self, x):
        if self.kernel_size == 1:
            return x
        pad_l = self.kernel_size - 1
        padding = (pad_l, 0)
        return F.pad(x, padding, mode="constant", value=0)

    def _same_padding(self, x):
        if self.kernel_size == 1:
            return x
        pad_total = self.kernel_size - 1
        pad_l = pad_total // 2
        pad_r = pad_total - pad_l
        padding = (pad_l, pad_r)
        return F.pad(x, padding, mode="constant", value=0)

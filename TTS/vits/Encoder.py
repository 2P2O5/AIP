from .utils import *

from .MultiHeadAttention import MultiHeadAttention
from .FFN import FFN
from .LayerNorm import LayerNorm
class Encoder(nn.Module):
    def __init__(
        self,
        v_DEVICE,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size=1,
        p_dropout=0.0,
        window_size=4,
        isflow=True,
        **kwargs
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.window_size = window_size
        self.cond_layer_idx = kwargs.get("cond_layer_idx", 2) if "gin_channels" in kwargs else self.n_layers
        if "gin_channels" in kwargs and kwargs["gin_channels"] != 0:
            self.gin_channels = kwargs["gin_channels"]
            self.spk_emb_linear = nn.Linear(self.gin_channels, self.hidden_channels)
            assert self.cond_layer_idx < self.n_layers, "cond_layer_idx should be less than n_layers"
        self.drop = nn.Dropout(p_dropout)
        self.attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()
        for i in range(self.n_layers):
            self.attn_layers.append(
                MultiHeadAttention(
                    v_DEVICE,
                    hidden_channels,
                    hidden_channels,
                    n_heads,
                    p_dropout=p_dropout,
                    window_size=window_size,
                )
            )
            self.norm_layers_1.append(LayerNorm(v_DEVICE, hidden_channels))
            self.ffn_layers.append(
                FFN(
                    v_DEVICE,
                    hidden_channels,
                    hidden_channels,
                    filter_channels,
                    kernel_size,
                    p_dropout=p_dropout,
                )
            )
            self.norm_layers_2.append(LayerNorm(v_DEVICE, hidden_channels))

    def forward(self, x, x_mask, g=None):
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x = x * x_mask
        g_transformed = None
        if g is not None and hasattr(self, 'spk_emb_linear'):
            g_transformed = self.spk_emb_linear(g.transpose(1, 2)).transpose(1, 2)
        
        for i, (attn_layer, norm1, ffn_layer, norm2) in enumerate(
            zip(self.attn_layers, self.norm_layers_1, self.ffn_layers, self.norm_layers_2)
        ):
            if i == self.cond_layer_idx and g_transformed is not None:
                x = (x + g_transformed) * x_mask
            y = attn_layer(x, x, attn_mask)
            y = self.drop(y)
            x = norm1(x + y)

            y = ffn_layer(x, x_mask)
            y = self.drop(y)
            x = norm2(x + y)
        return x * x_mask

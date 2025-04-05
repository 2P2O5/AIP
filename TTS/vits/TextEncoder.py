from .utils import *

from .Encoder import Encoder

from ..__symbols__ import *

class TextEncoder(nn.Module):
    def __init__(
        self,
        v_DEVICE,
        n_vocab,
        out_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        gin_channels=0,
    ):
        super().__init__()
        self.n_vocab = n_vocab
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels
        self.emb = nn.Embedding(len(symbols), hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)
        self.tone_emb = nn.Embedding(num_tones, hidden_channels)
        nn.init.normal_(self.tone_emb.weight, 0.0, hidden_channels**-0.5)
        self.language_emb = nn.Embedding(num_languages, hidden_channels)
        nn.init.normal_(self.language_emb.weight, 0.0, hidden_channels**-0.5)
        self.bert_proj = nn.Conv1d(1024, hidden_channels, 1)

        self.encoder = Encoder(
            v_DEVICE,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            gin_channels=self.gin_channels,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

        
        self.v_DEVICE = v_DEVICE

    def forward(self, x, x_lengths, tone, language, bert, g=None):
        x = (
            self.emb(x)
            + self.tone_emb(tone)
            + self.language_emb(language)
            + self.bert_proj(bert).transpose(1, 2)
        ) * math.sqrt(
            self.hidden_channels
        )  
        x = torch.transpose(x, 1, -1)  
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2), self.v_DEVICE), 1).to(
            x.dtype
        )

        x = self.encoder.forward(x * x_mask, x_mask, g=g) 
        stats = self.proj(x) * x_mask

        m, logs = torch.split(stats, self.out_channels, dim=1)
        return x, m, logs, x_mask

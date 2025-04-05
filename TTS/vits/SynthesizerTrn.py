from .utils import *

from .TextEncoder import TextEncoder
from .Generator import Generator
from .StochasticDurationPredictor import StochasticDurationPredictor
from .DurationPredictor import DurationPredictor
from .TransformerCouplingBlock import TransformerCouplingBlock

class SynthesizerTrn(nn.Module):
    """
    Synthesizer for Training
    """

    def __init__(
        self,
        v_DEVICE,
        n_vocab,
        spec_channels,
        segment_size,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        n_speakers=256,
        gin_channels=256,
        use_sdp=True,
        n_flow_layer=4,
        n_layers_trans_flow=3,
        flow_share_parameter=False,
        use_transformer_flow=True,
        **kwargs
    ):

        super().__init__()
        self.n_vocab = n_vocab
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels
        self.n_layers_trans_flow = n_layers_trans_flow
        self.use_spk_conditioned_encoder = kwargs.get(
            "use_spk_conditioned_encoder", True
        )
        self.use_sdp = use_sdp
        self.use_noise_scaled_mas = kwargs.get("use_noise_scaled_mas", False)
        self.mas_noise_scale_initial = kwargs.get("mas_noise_scale_initial", 0.01)
        self.noise_scale_delta = kwargs.get("noise_scale_delta", 2e-6)
        self.current_mas_noise_scale = self.mas_noise_scale_initial
        if self.use_spk_conditioned_encoder and gin_channels > 0:
            self.enc_gin_channels = gin_channels
        self.enc_p = TextEncoder(
            v_DEVICE,
            n_vocab,
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            gin_channels=self.enc_gin_channels,
        )
        self.dec = Generator(
            v_DEVICE,
            inter_channels,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=gin_channels,
        )
        if use_transformer_flow:
            self.flow = TransformerCouplingBlock(
                v_DEVICE,
                inter_channels,
                hidden_channels,
                filter_channels,
                n_heads,
                n_layers_trans_flow,
                5,
                p_dropout,
                n_flow_layer,
                gin_channels=gin_channels,
                share_parameter=flow_share_parameter,
            )
        else:
            print("Don't All use transformer_flow.")
            exit(1)
        self.sdp = StochasticDurationPredictor(
            v_DEVICE, hidden_channels, 192, 3, 0.5, 4, gin_channels=gin_channels
        )
        self.dp = DurationPredictor(
            v_DEVICE, hidden_channels, 256, 3, 0.5, gin_channels=gin_channels
        )

        self.emb_g = nn.Embedding(n_speakers, gin_channels)

        
        self.v_DEVICE = v_DEVICE

    def infer(
        self,
        x,
        x_lengths,
        sid,
        tone,
        language,
        bert,
        noise_scale=0.333,
        length_scale=1,
        noise_scale_w=0.333,
        sdp_ratio=0
    ):
        if self.n_speakers > 0:
            g = self.emb_g(sid).unsqueeze(-1)  
        else:
            g = None

        x, m_p, logs_p, x_mask = self.enc_p.forward(x, x_lengths, tone, language, bert, g=g)

        sdp_output = self.sdp.forward(x, x_mask, g=g, reverse=True, noise_scale=noise_scale_w)
        dp_output = self.dp.forward(x, x_mask, g=g)
        w_ceil = torch.ceil(torch.exp(sdp_output * sdp_ratio + dp_output * (1 - sdp_ratio)) * x_mask * length_scale)

        y_lengths = torch.sum(w_ceil, [1, 2]).clamp_min(1).long()
        y_mask = sequence_mask(y_lengths, None, self.v_DEVICE).unsqueeze(1).to(x_mask.dtype)

        attn_mask = x_mask.unsqueeze(2) * y_mask.unsqueeze(-1)
        b, _, t_y, t_x = attn_mask.shape
        cum_duration = torch.cumsum(w_ceil, -1)

        cum_duration_flat = cum_duration.view(b * t_x)
        attn = sequence_mask(cum_duration_flat, t_y, self.v_DEVICE).to(attn_mask.dtype)
        attn = attn.view(b, t_x, t_y)
        attn = attn - F.pad(attn, convert_pad_shape([[0, 0], [1, 0], [0, 0]]))[:, :-1]
        attn = attn.unsqueeze(1).transpose(2, 3) * attn_mask

        attn_squeezed = attn.squeeze(1)
        m_p = torch.bmm(attn_squeezed, m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.bmm(attn_squeezed, logs_p.transpose(1, 2)).transpose(1, 2)

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = self.flow.forward(z_p, y_mask, g=g, reverse=True)
        o = self.dec.forward(z * y_mask, g=g)
        return o

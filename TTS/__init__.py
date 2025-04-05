import torch
import numpy as np
import io
import wave
from .bert import LoadBert, get_text
from .vits import LoadTTS

DEVICE = torch.device("cpu")

def init(bert_path: str, tts_pth_path: str, tts_config_path: str = None, device: str = "cpu"):
    net_g, hparams = LoadTTS(tts_pth_path, tts_config_path, device)
    LoadBert(bert_path, device)
    DEVICE = torch.device(device)

    def _infer(text: str, sid: str, sdp_ratio=0, noise_scale=0.667, noise_scale_w=0.8, length_scale=1):
        bert, phones, tones, lang_ids = get_text(text, "ZH", hparams)

        x_tst = phones.to(DEVICE).unsqueeze(0)
        tones = tones.to(DEVICE).unsqueeze(0)
        lang_ids = lang_ids.to(DEVICE).unsqueeze(0)
        bert = bert.unsqueeze(0)
        x_tst_lengths = torch.LongTensor([phones.size(0)]).to(DEVICE)
        
        del phones
        speakers = torch.LongTensor([hparams.data.spk2id[sid]]).to(DEVICE)
        audio = (
            net_g.infer(
                x_tst,
                x_tst_lengths,
                speakers,
                tones,
                lang_ids,
                bert,
                sdp_ratio=sdp_ratio,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                length_scale=length_scale,
            )[0, 0].data.cpu().numpy()
        )
        del x_tst, tones, lang_ids, bert, x_tst_lengths, speakers

        audio_array = np.int16(audio / np.max(np.abs(audio)) * 32767)

        byte_io = io.BytesIO()
        with wave.open(byte_io, 'wb') as wave_out:
            wave_out.setnchannels(1)
            wave_out.setsampwidth(2)
            wave_out.setframerate(44100)
            audio_array = audio_array.astype(np.int16)
            wave_out.writeframes(audio_array.tobytes())
        return byte_io.getvalue()
    
    return _infer
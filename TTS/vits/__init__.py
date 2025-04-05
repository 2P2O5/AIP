from .SynthesizerTrn import SynthesizerTrn
import json
import os
from ..__symbols__ import symbols
from .utils import *

class HParams:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()
    
def _load_checkpoint(checkpoint_path, model, optimizer=None, skip_optimizer=False):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location=v_DEVICE)
    iteration = checkpoint_dict["iteration"]
    learning_rate = checkpoint_dict["learning_rate"]

    if optimizer and not skip_optimizer and checkpoint_dict.get("optimizer"):
        optimizer.load_state_dict(checkpoint_dict["optimizer"])
    elif optimizer and not skip_optimizer:
        new_opt_dict = optimizer.state_dict()
        new_opt_dict["param_groups"] = checkpoint_dict["optimizer"]["param_groups"]
        new_opt_dict["param_groups"][0]["params"] = optimizer.state_dict()["param_groups"][0]["params"]
        optimizer.load_state_dict(new_opt_dict)

    saved_state_dict = checkpoint_dict["model"]
    model_state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()

    new_state_dict = {
        k: (saved_state_dict[k] if k in saved_state_dict and saved_state_dict[k].shape == v.shape else v)
        for k, v in model_state_dict.items()
    }

    if hasattr(model, "module"):
        model.module.load_state_dict(new_state_dict, strict=False)
    else:
        model.load_state_dict(new_state_dict, strict=False)

    return model, optimizer, learning_rate, iteration


v_DEVICE = torch.device("cpu")
def LoadTTS(file, config_file=None, device="cpu"):
    """
    Load TTS model from file.
    :param file: path to the model file
    :param config_file: path to the config file
    :return: (TTS infer model, hparams)
    """
    global v_DEVICE
    v_DEVICE = torch.device(device)

    config_file = config_file or file.replace('.pth', '.json')

    with open(config_file, 'r', encoding="utf-8") as f:
        config = json.load(f)

    hparams = HParams(**config)
    net_g = SynthesizerTrn(
        v_DEVICE,
        len(symbols),
        hparams.data.filter_length // 2 + 1,
        hparams.train.segment_size // hparams.data.hop_length,
        n_speakers=hparams.data.n_speakers,
        **hparams.model,
    )

    net_g.eval()
    _load_checkpoint(file, net_g, None, skip_optimizer=True)
    net_g.to(v_DEVICE)

    return net_g, hparams
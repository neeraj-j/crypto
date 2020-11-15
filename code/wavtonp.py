#This file converts wav file to numpy array.

import soundfile as sf
import sounddevice as sd
import torch
import torch.nn.functional as F

wav_file = "../../../yoga/data/demo/fitness/sounds/1.wav"

def postprocess(feats, curr_sample_rate):
    if feats.ndim == 2:
        feats = feats.mean(-1)

    assert feats.ndim == 1, feats.ndim
    feats = torch.tensor(feats)
    with torch.no_grad():
         feats = F.layer_norm(feats, feats.shape)
    feats = feats.numpy()
    return feats

data, sample_rate = sf.read(wav_file)
data = postprocess(data, sample_rate)
sd.play(data,sample_rate)
status = sd.wait()
print(data)
print("")


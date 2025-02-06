import torch
from torch import Tensor
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

# model_name = "facebook/wav2vec2-large-xlsr-53"
model_name = "nguyenvulebinh/wav2vec2-base-vietnamese-250h"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
model = Wav2Vec2Model.from_pretrained(model_name)


def wav2vec2(waveform: Tensor):
    i = feature_extractor(waveform, return_tensors="pt", sampling_rate=16000)
    with torch.no_grad():
        o = model(i.input_values)
    return o.extract_features.mean(dim=-2)

import torch
import random
import numpy as np
import yaml
from munch import Munch
import torchaudio
import librosa
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel
import uvicorn
import io
from scipy.io import wavfile
import soundfile as sf
import onnxruntime as ort

# Set random seeds for reproducibility
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
random.seed(0)
np.random.seed(0)

# Add StyleTTS2 path
import sys
import os
import time

# Import necessary modules
from models import *
from utils import *



PAD = '_'
BOS = '<bos>'
EOS = '<eos>'
PUNC = '!?\'\"().,-=:;^&*~'
SPACE = ' '
_SILENCES = ['sp', 'spn', 'sil']

JAMO_LEADS = "".join([chr(_) for _ in range(0x1100, 0x1113)])
JAMO_VOWELS = "".join([chr(_) for _ in range(0x1161, 0x1176)])
JAMO_TAILS = "".join([chr(_) for _ in range(0x11A8, 0x11C3)])

VALID_CHARS = JAMO_LEADS + JAMO_VOWELS + JAMO_TAILS + PUNC + SPACE
symbols = [PAD] + [BOS] + [EOS] + list(VALID_CHARS) + _SILENCES

id_to_sym = {i: sym for i, sym in enumerate(symbols)}
#---
dicts = {}
for i in range(len((symbols))):
    dicts[symbols[i]] = i

from g2pK.g2pkc import G2p

g2pk = G2p()
class TextCleaner:
    def __init__(self, dummy=None):
        self.word_index_dictionary = dicts
    def __call__(self, text, cleaned=False):
        indexes = []
        if not cleaned:
            text = g2pk(text)
        for char in text:
            try:
                indexes.append(self.word_index_dictionary[char])
            except KeyError:
                print(text)
        return indexes

textcleaner = TextCleaner()

# Initialize device and other components
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Load ONNX model
opset = 16
text_encoder_onnx_model_path = f'Models/dani_onnx/text_encoder_dani_onnx_opset{opset}.onnx'
prosodic_text_encoder_onnx_model_path = f'Models/dani_onnx/prosodic_text_encoder_dani_onnx_opset{opset}.onnx'
diffusion_onnx_model_path = f'Models/dani_onnx/denoising_sampler_dani_step10_onnx_opset{opset}.onnx'
duration_predictor_onnx_model_path = f'Models/dani_onnx/duration_predictor_dani_onnx_opset{opset}.onnx'
prosody_predictor_onnx_model_path = f'Models/dani_onnx/prosody_predictor_dani_onnx_opset{opset}.onnx'
decoder_onnx_model_path = f'Models/dani_onnx/decoder_dani_onnx_opset{opset}.onnx'

# Create a session with the onnx model
text_encoder_ort_session = ort.InferenceSession(text_encoder_onnx_model_path, providers=[('CUDAExecutionProvider', {'cudnn_conv_algo_search': 'DEFAULT'}), 'CPUExecutionProvider'])
prosodic_text_encoder_ort_session = ort.InferenceSession(prosodic_text_encoder_onnx_model_path, providers=[('CUDAExecutionProvider', {'cudnn_conv_algo_search': 'DEFAULT'}), 'CPUExecutionProvider'])
diffusion_ort_session = ort.InferenceSession(diffusion_onnx_model_path, providers=[('CUDAExecutionProvider', {'cudnn_conv_algo_search': 'DEFAULT'}), 'CPUExecutionProvider'])
duration_predictor_ort_session = ort.InferenceSession(duration_predictor_onnx_model_path, providers=[('CUDAExecutionProvider', {'cudnn_conv_algo_search': 'DEFAULT'}), 'CPUExecutionProvider'])
prosody_predictor_ort_session = ort.InferenceSession(prosody_predictor_onnx_model_path, providers=[('CUDAExecutionProvider', {'cudnn_conv_algo_search': 'DEFAULT'}), 'CPUExecutionProvider'])
decoder_ort_session = ort.InferenceSession(
    decoder_onnx_model_path, 
    providers=[
            ('CUDAExecutionProvider', {'cudnn_conv_algo_search': 'DEFAULT'}),
            'CPUExecutionProvider'
            ], 
    )

ref_s_path = 'Models/dani_onnx/ref_dani.npy'
features = np.load(ref_s_path)
features = np.repeat(features, 1, axis=0)

# Helper functions
def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask


# Inference function
def inference(text):
    text = text.strip()
    text = text.replace('"', '')
    if not text:
        text = '음성이 제대로 인식되지 않았습니다.'
    token = textcleaner(text)
    token.insert(0, 0)
    token.append(0)
    token = [token]

    text_encoder_input = {
        'text': token,
    }
    t_en = text_encoder_ort_session.run(None, text_encoder_input)[0]

    prosodic_text_encoder_inputs = {
        'text': token,
    }
    d_en = prosodic_text_encoder_ort_session.run(None, prosodic_text_encoder_inputs)[0]

    noise = np.random.randn(1, 1, 256).astype(np.float32)
    embedding = d_en
    diffusion_inputs = {
        'noise': noise,
        'embedding': embedding,
        'features': features
    }
    s_pred = diffusion_ort_session.run(None, diffusion_inputs)[0].squeeze(1)

    style_a = s_pred[:, :128]
    style_p = s_pred[:, 128:]
    duration_predictor_inputs = {
        'text_emb': d_en.swapaxes(-1, -2),
        'style': style_p,
    }
    d, pred_dur = duration_predictor_ort_session.run(None, duration_predictor_inputs)

    frame_len = int(pred_dur.sum())
    aln_trg = np.zeros((len(token[0]), frame_len), dtype=np.int8)
    c_frame = 0
    for i in range(aln_trg.shape[0]):
        dur = int(pred_dur[i])
        aln_trg[i, c_frame:c_frame + dur] = 1
        c_frame += dur
    en = d.swapaxes(-1, -2) @ aln_trg

    prosody_predictor_inputs = {
        'en': en,
        'style': style_p,
    }
    F0, N = prosody_predictor_ort_session.run(None, prosody_predictor_inputs)

    en_decoder = t_en.swapaxes(-1, -2) @ aln_trg
    decoder_inputs = {
        'en': en_decoder,
        'F0': F0,
        'N': N,
        'style': style_a,
    }
    output = decoder_ort_session.run(None, decoder_inputs)[0].squeeze()

    return output


class TTSRequest(BaseModel):
    text: str

def text_to_speech(request: TTSRequest):
    # Perform
    start_time = time.time()
    wav = inference(request.text)
    print(f"Execution time: {time.time() - start_time} seconds")

    # Ensure the audio is in the correct range (-1 to 1)
    wav = np.clip(wav, -1, 1)

    # Print debug information
    print(f"Audio shape: {wav.shape}")
    print(f"Audio min: {wav.min()}, max: {wav.max()}")
    print(f"Intended sample rate: 24000")
                
    # save the audio to a file
    sf.write("output.wav", wav, 24000, format='wav')

if __name__ == "__main__":
    text_to_speech(TTSRequest(text="안녕하세요, 이것은 샘플 목소리입니다. 잘 들리시나요? 잘 들리신다면 잘 들리신다고 말씀해주세요!"))
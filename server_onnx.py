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

# ort.set_default_logger_severity(3)

# Set random seeds for reproducibility
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
random.seed(0)
np.random.seed(0)

# Add StyleTTS2 path
import sys
import os

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

# Inference function
def inference(token):
    # text = text.strip()
    # text = text.replace('"', '')
    # print(f"Input Text: {text}")
    # token = textcleaner(text)
    # token.insert(0, 0)
    # token.append(0)
    # token = [token]

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

    # Resample to 16kHz
    original_sample_rate = 24000
    target_sample_rate = 16000
    waveform = torch.tensor(output).unsqueeze(0)  # Add batch dimension for torchaudio
    resampled_waveform = torchaudio.transforms.Resample(
        orig_freq=original_sample_rate, new_freq=target_sample_rate
    )(waveform)
    resampled_waveform = resampled_waveform.squeeze(0).numpy()  # Remove batch dimension

    return resampled_waveform

# FastAPI app
app = FastAPI()

class TTSRequest(BaseModel):
    text: str


import re

ENDOFSENTENCE_PATTERN_STR = r"""
    (?<![A-Z])       # Negative lookbehind: not preceded by an uppercase letter (e.g., "U.S.A.")
    (?<!\d)          # Negative lookbehind: not preceded by a digit (e.g., "1. Let's start")
    (?<!\d\s[ap])    # Negative lookbehind: not preceded by time (e.g., "3:00 a.m.")
    (?<!Mr|Ms|Dr)    # Negative lookbehind: not preceded by Mr, Ms, Dr (combined bc. length is the same)
    (?<!Mrs)         # Negative lookbehind: not preceded by "Mrs"
    (?<!Prof)        # Negative lookbehind: not preceded by "Prof"
    [\.\?\!:;]|      # Match a period, question mark, exclamation point, colon, or semicolon
    [。？！：；]       # the full-width version (mainly used in East Asian languages such as Chinese)
    $                # End of string
"""
ENDOFSENTENCE_PATTERN = re.compile(ENDOFSENTENCE_PATTERN_STR, re.VERBOSE)


def match_endofsentence(text: str) -> int:
    match = ENDOFSENTENCE_PATTERN.search(text.rstrip())
    return match.end() if match else 0

# 문장 누적 및 처리 플래그
AGGREGATE_SENTENCES = True
current_sentence = ""  # 전역(또는 싱글톤/클래스 멤버 등)으로 문장 누적 버퍼 하나를 둠

@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    try:
        global current_sentence
        # 텍스트가 비어있는지 체크
        text_input = request.text
        text_input = text_input.strip()
        text_input = text_input.replace('"', '')

        # if AGGREGATE_SENTENCES:
        #     current_sentence += text_input
        #     eos_end_marker = match_endofsentence(current_sentence)
        #     if eos_end_marker:
        #         text_to_infer = current_sentence[:eos_end_marker]
        #         current_sentence = current_sentence[eos_end_marker:]
        #     else:
        #         print("=== Waiting for more text to infer ===")
        #         print(current_sentence)
        #         print("=====================================")
        #         return None
        # else:  
        #     text_to_infer = text_input

        print("--------------------")
        print(f"Input Text: {text_input}")
        print("--------------------")
        token = textcleaner(text_input)
        if len(token) <= 3: 
            # raise HTTPException(status_code=422, detail="Input text is too short or invalid.")
            return None
        
        token.insert(0, 0)
        token.append(0)
        token = [token]
        wav_float = inference(token)

        sample_rate = 24000
        wav_int16 = np.clip(wav_float, -1, 1)
        wav_int16 = (wav_int16 * 32767).astype(np.int16)

        pcm_bytes = wav_int16.tobytes()

        return StreamingResponse(
            content=io.BytesIO(pcm_bytes), 
            media_type="audio/l16; rate=24000; channels=1")

    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8014)

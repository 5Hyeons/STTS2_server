import torch
import random
import numpy as np
import yaml
from munch import Munch
import torchaudio
import librosa
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
import io
from scipy.io import wavfile
import soundfile as sf

# Set random seeds for reproducibility
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
random.seed(0)
np.random.seed(0)

# Add StyleTTS2 path
import sys
import os
styletts2_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'StyleTTS2'))
sys.path.insert(0, styletts2_path)
print(styletts2_path)

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

# Initialize mel spectrogram transform
to_mel = torchaudio.transforms.MelSpectrogram(n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
mean, std = -4, 4

# Load configuration
config = yaml.safe_load(open("Models/dani/config_ft_hh.yml"))

# Load models
ASR_config = config.get('ASR_config', False)
ASR_path = config.get('ASR_path', False)
text_aligner = load_ASR_models(ASR_path, ASR_config)


F0_path = config.get('F0_path', False)
pitch_extractor = load_F0_models(F0_path)

from Utils.PLBERT.util import load_plbert
BERT_path = config.get('PLBERT_dir', False)
plbert = load_plbert(BERT_path)

model = build_model(recursive_munch(config['model_params']), text_aligner, pitch_extractor, plbert)
_ = [model[key].eval() for key in model]
_ = [model[key].to(device) for key in model]

# Load model parameters
params_whole = torch.load("Models/dani/epoch_2nd_00249.pth", map_location='cpu')
params = params_whole['net']

for key in model:
    if key in params:
        print('%s loaded' % key)
        try:
            model[key].load_state_dict(params[key])
        except:
            from collections import OrderedDict
            state_dict = params[key]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            model[key].load_state_dict(new_state_dict, strict=False)

_ = [model[key].eval() for key in model]

# Initialize sampler
from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule

sampler = DiffusionSampler(
    model.diffusion.diffusion,
    sampler=ADPM2Sampler(),
    sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
    clamp=False
)

# Helper functions
def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask

# Inference function
def preprocess(wave):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor

def compute_style(path):
    audio, sr = librosa.load(path, sr=24000)
    if sr != 24000:
        audio = librosa.resample(audio, sr, 24000)
    mel_tensor = preprocess(audio).to(device)

    with torch.no_grad():
        ref_s = model.style_encoder(mel_tensor.unsqueeze(1))
        ref_p = model.predictor_encoder(mel_tensor.unsqueeze(1))

    return torch.cat([ref_s, ref_p], dim=1)


ref_wav_path = "Models/dani/fairy_0012.wav"
ref_s = compute_style(ref_wav_path)

def inference(text, alpha = 0.3, beta = 0.7, diffusion_steps=15, embedding_scale=1):
    text = text.strip()
    tokens = textcleaner(text)
    tokens.insert(0, 0)
    tokens.append(0)
    tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)
    
    with torch.no_grad():
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
        text_mask = length_to_mask(input_lengths).to(device)

        t_en = model.text_encoder(tokens, input_lengths, text_mask)
        d_en = model.prosodic_text_encoder(tokens, input_lengths, text_mask)
        d_en_dur = d_en.transpose(-1, -2)

        s_pred = sampler(noise = torch.randn((1, 256)).unsqueeze(1).to(device), 
                                          embedding=d_en_dur,
                                          embedding_scale=embedding_scale,
                                            features=ref_s, # reference from the same speaker as the embedding
                                             num_steps=diffusion_steps).squeeze(1)

        s = s_pred[:, 128:]
        ref = s_pred[:, :128]

        ref = alpha * ref + (1 - alpha)  * ref_s[:, :128]
        s = beta * s + (1 - beta)  * ref_s[:, 128:]

        d = model.predictor.text_encoder(d_en, 
                                         s, input_lengths, text_mask)

        x, _ = model.predictor.lstm(d)
        duration = model.predictor.duration_proj(x)

        duration = torch.sigmoid(duration).sum(axis=-1)
        pred_dur = torch.round(duration.squeeze()).clamp(min=1)


        pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
        c_frame = 0
        for i in range(pred_aln_trg.size(0)):
            pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
            c_frame += int(pred_dur[i].data)

        # encode prosody
        en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device))
        asr_new = torch.zeros_like(en)
        asr_new[:, :, 0] = en[:, :, 0]
        asr_new[:, :, 1:] = en[:, :, 0:-1]
        en = asr_new
        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)

        asr = (t_en @ pred_aln_trg.unsqueeze(0).to(device))
        asr_new = torch.zeros_like(asr)
        asr_new[:, :, 0] = asr[:, :, 0]
        asr_new[:, :, 1:] = asr[:, :, 0:-1]
        asr = asr_new

        out = model.decoder(asr, 
                                F0_pred, N_pred, ref.squeeze().unsqueeze(0))
        
    return out.squeeze().cpu().numpy()[..., :-50] # weird pulse at the end of the model, need to be fixed later

# FastAPI app
app = FastAPI()

class TTSRequest(BaseModel):
    text: str

@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    try:
        wav = inference(request.text)

        # Ensure the audio is in the correct range (-1 to 1)
        wav = np.clip(wav, -1, 1)

        # Print debug information
        print(f"Audio shape: {wav.shape}")
        print(f"Audio min: {wav.min()}, max: {wav.max()}")
        print(f"Intended sample rate: 24000")
                                        
                        
        # Convert to bytes using an in-memory buffer
        buffer = io.BytesIO()
        sf.write(buffer, wav, 24000, format='wav')
        buffer.seek(0)

        # Read back the file to check its properties
        with sf.SoundFile(buffer) as sf_file:
            print(f"Actual sample rate: {sf_file.samplerate}")
            print(f"Channels: {sf_file.channels}")
            print(f"Format: {sf_file.format}")
            print(f"Subtype: {sf_file.subtype}")

        return StreamingResponse(content=buffer, media_type="audio/wav")
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8014)

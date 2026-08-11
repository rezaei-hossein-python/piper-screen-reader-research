import sys
import os
from pathlib import Path
import wave

# Add upstream piper source
sys.path.insert(0, 'C:/projects/piper-screen-reader-research/upstream/piper/src')
from piper.voice import PiperVoice
from piper.config import SynthesisConfig

model_path = 'C:/projects/nvda piper addon/.phase2h-assets/en_US-lessac-low/en_US-lessac-low.onnx'
voice = PiperVoice.load(model_path)

dataset_dir = Path('C:/projects/piper-screen-reader-research/training/dataset')
wavs_dir = dataset_dir / 'wavs'
wavs_dir.mkdir(parents=True, exist_ok=True)

tokens = ["A", "E", "F", "K", "R", "S", "U", "W", "0", "5", "7", "button", "selected", "expanded", "unavailable"]

metadata_lines = []

def save_wav(path, text, length_scale):
    syn_config = SynthesisConfig(length_scale=length_scale)
    wav_file = wave.open(str(path), "wb")
    wav_params_set = False
    with wav_file:
        for i, audio_chunk in enumerate(voice.synthesize(text, syn_config)):
            if not wav_params_set:
                wav_file.setframerate(audio_chunk.sample_rate)
                wav_file.setsampwidth(audio_chunk.sample_width)
                wav_file.setnchannels(audio_chunk.sample_channels)
                wav_params_set = True
            wav_file.writeframes(audio_chunk.audio_int16_bytes)

for token in tokens:
    filename_token = token
    if token == "0": filename_token = "zero"
    elif token == "5": filename_token = "five"
    elif token == "7": filename_token = "seven"
    
    # 1. Normal Mode
    norm_path = wavs_dir / f"normal_{filename_token}.wav"
    save_wav(norm_path, token, length_scale=1.0)
    metadata_lines.append(f"wavs/normal_{filename_token}.wav|{token}|normal")
    print(f"Generated normal WAV for: {token}")
    
    # 2. Interactive Mode
    int_path = wavs_dir / f"interactive_{filename_token}.wav"
    save_wav(int_path, token, length_scale=0.5)
    metadata_lines.append(f"wavs/interactive_{filename_token}.wav|{token}|interactive")
    print(f"Generated interactive WAV for: {token}")

# Save metadata CSV
csv_path = dataset_dir / "metadata.csv"
csv_path.write_text("\n".join(metadata_lines) + "\n", encoding="utf-8")
print(f"Generated metadata.csv at {csv_path}")

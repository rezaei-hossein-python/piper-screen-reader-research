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

# 8 microscopic representative difficult items
tokens = ["F", "N", "m", "b", "V", "list", "link", "comma"]

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

print("Generating Stage 1 microscopic dataset WAVs at 16,000 Hz...")
for token in tokens:
    filename_token = token
    # Lowercase or clean up name for filename if necessary
    if token == "F": filename_token = "F"
    elif token == "N": filename_token = "N"
    elif token == "m": filename_token = "m"
    elif token == "b": filename_token = "b"
    elif token == "V": filename_token = "V"
    
    # 1. Normal Mode (length_scale = 1.0)
    norm_path = wavs_dir / f"normal_{filename_token}.wav"
    # To be safe, generate if not exists, or overwrite to ensure exact 16kHz
    save_wav(norm_path, token, length_scale=1.0)
    metadata_lines.append(f"wavs/normal_{filename_token}.wav|{token}|normal")
    print(f"  Generated normal WAV for: {token} -> {norm_path.name}")
    
    # 2. Interactive Mode (length_scale = 0.5)
    int_path = wavs_dir / f"interactive_{filename_token}.wav"
    save_wav(int_path, token, length_scale=0.5)
    metadata_lines.append(f"wavs/interactive_{filename_token}.wav|{token}|interactive")
    print(f"  Generated interactive WAV for: {token} -> {int_path.name}")

# Save Stage 1 metadata CSV
csv_path = dataset_dir / "metadata_stage1.csv"
csv_path.write_text("\n".join(metadata_lines) + "\n", encoding="utf-8")
print(f"Generated Stage 1 metadata_stage1.csv at {csv_path}")

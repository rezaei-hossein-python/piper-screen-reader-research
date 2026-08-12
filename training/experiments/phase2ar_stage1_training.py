import sys
import os
import time
import json
import wave
import pathlib
from pathlib import Path
from unittest.mock import MagicMock
import torch
import torch.nn.functional as F
import numpy as np

# Fix pathlib PosixPath cross-platform instantiation issue on Windows
pathlib.PosixPath = pathlib.WindowsPath
torch.serialization.add_safe_globals([pathlib.PosixPath, pathlib.WindowsPath])

# Create a mock for monotonic_align so we can run forward-pass training loops on CPU
mock_mono = MagicMock()
def mock_max_path(neg_cent, mask):
    B, T_t, T_s = neg_cent.shape
    attn = torch.zeros_like(neg_cent)
    for b in range(B):
        for t in range(T_t):
            s = int(t * T_s / T_t)
            attn[b, t, min(s, T_s - 1)] = 1.0
    return attn * mask

mock_mono.maximum_path = mock_max_path
sys.modules['piper.train.vits.monotonic_align'] = mock_mono

UPSTREAM_PATH = Path("C:/projects/piper-screen-reader-research/upstream/piper/src")
sys.path.insert(0, str(UPSTREAM_PATH))

from piper.train.vits.lightning import VitsModel
from piper.train.vits.dataset import VitsDataModule
from piper.train.vits.commons import slice_segments
from piper.train.vits.mel_processing import mel_spectrogram_torch, spec_to_mel_torch

def compute_mel_spectrogram(audio, hparams):
    # Computes mel spectrogram from raw audio waveform
    return mel_spectrogram_torch(
        audio.squeeze(1).float(),
        hparams.filter_length,
        hparams.mel_channels,
        hparams.sample_rate,
        hparams.hop_length,
        hparams.win_length,
        hparams.mel_fmin,
        hparams.mel_fmax,
    )

def run_stage1_training():
    print("=== Phase 2AR: Stage 1 Training Execution ===")
    
    # Paths setup
    base_ckpt_path = "C:/projects/piper-screen-reader-research/models/lessac/epoch=2307-step=558536.ckpt"
    results_dir = Path("C:/projects/piper-screen-reader-research/training/results/phase2ar")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load model from base checkpoint
    print(f"Loading baseline checkpoint: {base_ckpt_path}")
    model = VitsModel.load_from_checkpoint(
        base_ckpt_path,
        strict=False,
        gin_channels=256,
        weights_only=False
    )
    print("Base model loaded successfully!")
    
    # 2. Setup zero-locks
    torch.nn.init.zeros_(model.model_g.emb_mode.weight[0])
    if model.model_g.dec.cond.bias is not None:
        model.model_g.dec.cond.bias.data.zero_()
        model.model_g.dec.cond.bias.requires_grad = False
    print("Normal mode zero-locks applied.")
    
    # 3. Freeze all parameters except emb_mode, dp.cond, and dec.cond
    for param in model.parameters():
        param.requires_grad = False
        
    trainable_names = ["emb_mode", "dp.cond", "dec.cond"]
    for name, param in model.named_parameters():
        if any(t in name for t in trainable_names):
            if "bias" not in name:
                param.requires_grad = True
                
    print("Trainable parameters unfrozen.")
    
    # Save Step 0 (initial) state
    step0_emb_mode = model.model_g.emb_mode.weight.clone().detach()
    step0_dp_cond = model.model_g.dp.cond.weight.clone().detach()
    step0_dec_cond = model.model_g.dec.cond.weight.clone().detach()
    
    # Initialize interactive parameters to break symmetry and allow adaptation
    with torch.no_grad():
        model.model_g.emb_mode.weight[1].normal_(mean=0.0, std=0.02)
        model.model_g.dec.cond.weight.normal_(mean=0.0, std=0.02)
        model.model_g.dp.cond.weight.normal_(mean=0.0, std=0.02)
        
    # 4. Set up VitsDataModule for Stage 1 microscopic items
    print("Setting up VitsDataModule with metadata_stage1.csv...")
    datamodule = VitsDataModule(
        csv_path="C:/projects/piper-screen-reader-research/training/dataset/metadata_stage1.csv",
        cache_dir="C:/projects/piper-screen-reader-research/training/dataset/cache_stage1",
        espeak_voice="en-US",
        config_path="C:/projects/nvda piper addon/.phase2h-assets/en_US-lessac-low/en_US-lessac-low.onnx.json",
        voice_name="lessac",
        batch_size=8,
        trim_silence=False,
    )
    datamodule.prepare_data()
    datamodule.setup(stage="fit")
    train_loader = datamodule.train_dataloader()
    print(f"Data module prepared! Batches in loader: {len(train_loader)}")
    
    # Setup evaluation corpus (the 8 Stage 1 items)
    eval_tokens = ["F", "N", "m", "b", "V", "list", "link", "comma"]
    
    # Helper to load target wav mel spectrogram
    target_mels = {}
    for token in eval_tokens:
        wav_path = Path(f"C:/projects/piper-screen-reader-research/training/dataset/wavs/interactive_{token}.wav")
        if wav_path.exists():
            # Load wave using wave module
            with wave.open(str(wav_path), "rb") as wf:
                params = wf.getparams()
                frames = wf.readframes(params.nframes)
                audio_np = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                audio_tensor = torch.FloatTensor(audio_np).unsqueeze(0).unsqueeze(0) # [1, 1, T]
                
                # Compute target mel spectrogram
                with torch.no_grad():
                    mel = compute_mel_spectrogram(audio_tensor, model.hparams)
                    target_mels[token] = mel
                    
    print(f"Loaded target mel spectrograms for {len(target_mels)} items.")
    
    # Voice configuration for phonemization during evaluation
    from piper.voice import PiperVoice
    voice = PiperVoice.load('C:/projects/nvda piper addon/.phase2h-assets/en_US-lessac-low/en_US-lessac-low.onnx')
    
    def evaluate_milestone(step_num, last_loss_dur, last_loss_mel):
        model.eval()
        results = []
        
        # Track gradient norms
        grad_emb = model.model_g.emb_mode.weight.grad.norm().item() if model.model_g.emb_mode.weight.grad is not None else 0.0
        grad_dp = model.model_g.dp.cond.weight.grad.norm().item() if model.model_g.dp.cond.weight.grad is not None else 0.0
        grad_dec = model.model_g.dec.cond.weight.grad.norm().item() if model.model_g.dec.cond.weight.grad is not None else 0.0
        
        normal_durations = []
        
        for token in eval_tokens:
            phonemes = voice.phonemize(token)[0]
            phoneme_ids = voice.phonemes_to_ids(phonemes)
            x_test = torch.LongTensor([phoneme_ids])
            x_lengths_test = torch.LongTensor([len(phoneme_ids)])
            
            # Modes
            mode_norm = torch.LongTensor([0])
            mode_int = torch.LongTensor([1])
            
            with torch.no_grad():
                # N0: Normal Mode
                o_norm, _, _, _ = model.model_g.infer(x_test, x_lengths_test, sid=None, speech_mode=mode_norm)
                norm_dur = (o_norm.size(-1) / 16000.0) * 1000.0
                normal_durations.append(norm_dur)
                
                # I2: Adapted Interactive Mode
                o_i2, _, _, _ = model.model_g.infer(x_test, x_lengths_test, sid=None, speech_mode=mode_int)
                i2_dur = (o_i2.size(-1) / 16000.0) * 1000.0
                
                # I1: Duration-Only Concise (by temporarily zeroing dec.cond.weight contribution)
                orig_dec_weight = model.model_g.dec.cond.weight.clone()
                model.model_g.dec.cond.weight.zero_()
                o_i1, _, _, _ = model.model_g.infer(x_test, x_lengths_test, sid=None, speech_mode=mode_int)
                i1_dur = (o_i1.size(-1) / 16000.0) * 1000.0
                model.model_g.dec.cond.weight.copy_(orig_dec_weight)
                
                # Calculate Mel L1 Spectral Difference comparing I1 vs target, and I2 vs target
                # Check if target mel exists
                if token in target_mels:
                    target_mel = target_mels[token]
                    
                    mel_i1 = compute_mel_spectrogram(o_i1, model.hparams)
                    mel_i2 = compute_mel_spectrogram(o_i2, model.hparams)
                    
                    # Trim/interpolate to compare Mel L1 loss
                    # We can use adaptive pooling or simple linear interpolation to match frames
                    min_frames = min(target_mel.size(-1), mel_i1.size(-1), mel_i2.size(-1))
                    t_mel = target_mel[..., :min_frames]
                    m_i1 = mel_i1[..., :min_frames]
                    m_i2 = mel_i2[..., :min_frames]
                    
                    l1_i1 = F.l1_loss(m_i1, t_mel).item()
                    l1_i2 = F.l1_loss(m_i2, t_mel).item()
                else:
                    l1_i1 = 0.0
                    l1_i2 = 0.0
                    
            results.append({
                "token": token,
                "n0_dur_ms": norm_dur,
                "i1_dur_ms": i1_dur,
                "i2_dur_ms": i2_dur,
                "l1_i1_vs_target": l1_i1,
                "l1_i2_vs_target": l1_i2,
                "clipping_i2": bool(o_i2.abs().max().item() > 1.0),
                "nan_inf_i2": bool(not torch.isfinite(o_i2).all())
            })
            
        model.train()
        
        # Calculate normal mode zero drift from step 0 baseline normal median
        avg_norm_dur = np.mean(normal_durations)
        
        print(f"\n--- Checkpoint Step {step_num} Evaluation ---")
        print(f"  Avg Duration Loss: {last_loss_dur:.4f}")
        print(f"  Avg Mel L1 Loss:   {last_loss_mel:.4f}")
        print(f"  Gradients: emb_mode={grad_emb:.4f} | dp.cond={grad_dp:.4f} | dec.cond={grad_dec:.4f}")
        print(f"  Normal Mode Avg Dur: {avg_norm_dur:.1f} ms")
        print(f"  Items Comparison:")
        for r in results:
            print(f"    Token: {r['token']:<6} | N0: {r['n0_dur_ms']:.1f} ms | I1: {r['i1_dur_ms']:.1f} ms | I2: {r['i2_dur_ms']:.1f} ms | L1 Diff (I1->I2): {r['l1_i1_vs_target']:.4f} -> {r['l1_i2_vs_target']:.4f}")
            
        return {
            "step": step_num,
            "duration_loss": last_loss_dur,
            "mel_loss": last_loss_mel,
            "grad_emb": grad_emb,
            "grad_dp": grad_dp,
            "grad_dec": grad_dec,
            "normal_avg_dur": avg_norm_dur,
            "items": results
        }

    # Optimizer
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
    
    milestones = [50, 100, 250]
    frontier_metrics = []
    
    # Helper to get batch generator
    def infinite_batches(loader):
        while True:
            for b in loader:
                yield b
                
    batch_gen = infinite_batches(train_loader)
    
    # Report Step 0 baseline before training
    model.train()
    metrics_step0 = evaluate_milestone(0, 0.0, 0.0)
    frontier_metrics.append(metrics_step0)
    
    current_step = 0
    print("\nStarting Stage 1 training loop on CPU...")
    for target_step in milestones:
        steps_to_run = target_step - current_step
        print(f"\nTraining for {steps_to_run} steps to reach step {target_step}...")
        
        dur_losses = []
        mel_losses = []
        
        for _ in range(steps_to_run):
            batch = next(batch_gen)
            
            x = batch.phoneme_ids
            x_lengths = batch.phoneme_lengths
            spec = batch.spectrograms
            spec_lengths = batch.spectrogram_lengths
            speaker_ids = batch.speaker_ids
            speech_modes = batch.speech_modes
            
            # Forward pass
            (
                y_hat,
                l_length,
                _attn,
                ids_slice,
                _x_mask,
                z_mask,
                (_z, z_p, m_p, logs_p, _m_q, logs_q),
            ) = model.model_g(
                x,
                x_lengths,
                spec,
                spec_lengths,
                speaker_ids,
                speech_mode=speech_modes,
            )
            
            # 1. Stochastic Duration Predictor Loss
            loss_dur = torch.sum(l_length.float())
            
            # 2. Mel reconstruction Loss
            # Compute target mel spectrogram segment
            mel = spec_to_mel_torch(
                spec.float(),
                model.hparams.filter_length,
                model.hparams.mel_channels,
                model.hparams.sample_rate,
                model.hparams.mel_fmin,
                model.hparams.mel_fmax,
            )
            y_mel = slice_segments(
                mel,
                ids_slice,
                model.hparams.segment_size // model.hparams.hop_length,
            )
            
            # Generated mel segment
            y_hat_mel = compute_mel_spectrogram(y_hat, model.hparams)
            
            # Mel spectrogram L1 reconstruction loss
            loss_mel = F.l1_loss(y_mel, y_hat_mel) * model.hparams.c_mel
            
            # Dual-signal total loss
            total_loss = loss_dur + loss_mel
            
            optimizer.zero_grad()
            total_loss.backward()
            
            # Protect normal mode
            if model.model_g.emb_mode.weight.grad is not None:
                model.model_g.emb_mode.weight.grad[0].zero_()
                
            optimizer.step()
            
            with torch.no_grad():
                model.model_g.emb_mode.weight[0].zero_()
                
            dur_losses.append(loss_dur.item())
            mel_losses.append(loss_mel.item())
            current_step += 1
            
        # Milestone reached! Save checkpoint and evaluate
        ckpt_path = results_dir / f"checkpoint_stage1_step_{current_step}.ckpt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"\nSaved Stage 1 checkpoint at step {current_step} -> {ckpt_path.name}")
        
        milestone_metrics = evaluate_milestone(current_step, np.mean(dur_losses), np.mean(mel_losses))
        frontier_metrics.append(milestone_metrics)
        
    # Save Stage 1 metrics JSON file
    metrics_path = results_dir / "stage1-results.json"
    metrics_path.write_text(json.dumps(frontier_metrics, indent=2))
    print(f"\nWritten Stage 1 metrics results to {metrics_path}")
    
    # 7. Generate Tiny Explicit Diagnostic Listening Set (5 items: F, N, b, list, comma)
    diag_tokens = ["F", "N", "b", "list", "comma"]
    diag_dir = results_dir / "diagnostic_set"
    diag_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating microscopic Stage 1 diagnostic listening set at {diag_dir}...")
    model.eval()
    
    for token in diag_tokens:
        phonemes = voice.phonemize(token)[0]
        phoneme_ids = voice.phonemes_to_ids(phonemes)
        x_test = torch.LongTensor([phoneme_ids])
        x_lengths_test = torch.LongTensor([len(phoneme_ids)])
        
        mode_norm = torch.LongTensor([0])
        mode_int = torch.LongTensor([1])
        
        with torch.no_grad():
            # N0: Normal Mode
            o_norm, _, _, _ = model.model_g.infer(x_test, x_lengths_test, sid=None, speech_mode=mode_norm)
            
            # I2: Adapted Interactive Mode
            o_i2, _, _, _ = model.model_g.infer(x_test, x_lengths_test, sid=None, speech_mode=mode_int)
            
            # I1: Duration-Only Concise (by zeroing dec.cond weight temporarily)
            orig_dec_weight = model.model_g.dec.cond.weight.clone()
            model.model_g.dec.cond.weight.zero_()
            o_i1, _, _, _ = model.model_g.infer(x_test, x_lengths_test, sid=None, speech_mode=mode_int)
            model.model_g.dec.cond.weight.copy_(orig_dec_weight)
            
        def save_waveform(path, waveform):
            scaled = waveform.clamp(-1.0, 1.0) * 32767.0
            scaled_bytes = scaled.short().cpu().numpy().tobytes()
            with wave.open(str(path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(scaled_bytes)
                
        # Use explicit, clean names so the user can directly compare
        save_waveform(diag_dir / f"eval_normal_{token}.wav", o_norm[0])
        save_waveform(diag_dir / f"eval_duration_only_{token}.wav", o_i1[0])
        save_waveform(diag_dir / f"eval_adapted_{token}.wav", o_i2[0])
        print(f"  Generated diagnostics for: {token}")
        
    print("\n=== STAGE 1 TRAINING AND DIAGNOSTICS GENERATION COMPLETE ===")
    
if __name__ == "__main__":
    run_stage1_training()

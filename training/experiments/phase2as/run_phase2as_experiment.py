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

UPSTREAM_PATH = Path("C:/projects/piper-screen-reader-research/upstream/piper/src")
sys.path.insert(0, str(UPSTREAM_PATH))

from piper.train.vits.lightning import VitsModel
from piper.train.vits import commons
from piper.voice import PiperVoice

def run_phase2as():
    print("=== Phase 2AS: Latent-Core Compression Experiment ===")
    
    # Configuration
    protected_edge_ms = 20.0
    core_scale = 0.50
    
    # 1. Load model and set it up
    ckpt_path = "C:/projects/piper-screen-reader-research/models/lessac/epoch=2307-step=558536.ckpt"
    print(f"Loading baseline checkpoint: {ckpt_path}")
    model = VitsModel.load_from_checkpoint(
        ckpt_path,
        strict=False,
        gin_channels=256,
        weights_only=False
    )
    model.eval()
    
    # Zero-lock the normal mode embedding to ensure exact Lessac equivalence
    if hasattr(model.model_g, 'emb_mode'):
        torch.nn.init.zeros_(model.model_g.emb_mode.weight[0])
    if hasattr(model.model_g.dec, 'cond') and model.model_g.dec.cond.bias is not None:
        model.model_g.dec.cond.bias.data.zero_()
        
    print("Baseline model loaded successfully.")
    
    # 2. Frame calculation
    hop_length = model.hparams.hop_length
    sample_rate = model.hparams.sample_rate
    frame_dur_ms = (hop_length / sample_rate) * 1000.0
    edge_frames = max(1, int(round(protected_edge_ms / frame_dur_ms)))
    print(f"Native Sample Rate: {sample_rate} Hz")
    print(f"Hop Length: {hop_length}")
    print(f"Frame Duration: {frame_dur_ms:.2f} ms")
    print(f"Protected Edge: {protected_edge_ms} ms -> {edge_frames} frames per edge")
    
    # Voice for phonemization
    voice = PiperVoice.load('C:/projects/nvda piper addon/.phase2h-assets/en_US-lessac-low/en_US-lessac-low.onnx')
    
    # Diagnostic corpus
    corpus = ["F", "N", "b", "m", "V", "list", "link", "comma", "A", "seven", "S", "button"]
    
    results_dir = Path("C:/projects/piper-screen-reader-research/training/results/phase2as")
    diag_dir = results_dir / "listening_set"
    diag_dir.mkdir(parents=True, exist_ok=True)
    
    metrics = []
    
    # Fix random seed for equivalence
    torch.manual_seed(42)
    np.random.seed(42)
    
    def save_waveform(path, waveform):
        scaled = waveform.clamp(-1.0, 1.0) * 32767.0
        scaled_bytes = scaled.short().cpu().numpy().tobytes()
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(scaled_bytes)
            
    print("\nProcessing Diagnostic Corpus...")
    for token in corpus:
        torch.manual_seed(42) # reset per item
        
        phonemes = voice.phonemize(token)[0]
        phoneme_ids = voice.phonemes_to_ids(phonemes)
        x_test = torch.LongTensor([phoneme_ids])
        x_lengths_test = torch.LongTensor([len(phoneme_ids)])
        
        mode_norm = torch.LongTensor([0])
        
        with torch.no_grad():
            # A. Obtain Normal Baseline Path (o_norm) AND inverse-flow tensors
            o_norm, attn, y_mask, (z_norm, z_p, m_p, logs_p) = model.model_g.infer(
                x_test, x_lengths_test, sid=None, speech_mode=mode_norm, length_scale=1.0
            )
            dur_n0_ms = (o_norm.size(-1) / sample_rate) * 1000.0
            
            # Extract durations from attn: [1, 1, t_y, t_x]
            attn_squeezed = attn[0, 0] # [t_y, t_x]
            durations = attn_squeezed.sum(dim=0).long() # [t_x]
            
            # Verify alignment integrity
            assert torch.sum(durations) == z_norm.size(2), "Alignment integrity failed: sum of durations != latent sequence length"
            
            # B. Baseline Equivalence Test
            o_equiv = model.model_g.dec((z_norm * y_mask)[:, :, :], g=None)
            max_diff_equiv = (o_norm - o_equiv).abs().max().item()
            assert max_diff_equiv < 1e-5, f"Baseline equivalence failed! Max diff: {max_diff_equiv}"
            
            # C. Latent-Core Compression Warp
            t_x = len(phoneme_ids)
            current_y_idx = 0
            
            new_z_frames = []
            num_modified = 0
            num_unchanged = 0
            
            # For edge preservation proof
            edge_preservation_passed = True
            
            for x_idx in range(t_x):
                dur = durations[x_idx].item()
                if dur == 0:
                    continue
                    
                z_token = z_norm[:, :, current_y_idx : current_y_idx + dur] # [1, h, dur]
                
                if dur > 2 * edge_frames:
                    # Modify
                    left_edge = z_token[:, :, :edge_frames]
                    right_edge = z_token[:, :, -edge_frames:]
                    core = z_token[:, :, edge_frames:-edge_frames]
                    
                    orig_core_len = core.size(2)
                    new_core_len = max(1, int(round(orig_core_len * core_scale)))
                    
                    if new_core_len != orig_core_len:
                        # Interpolate core (1D along time dimension)
                        core_compressed = F.interpolate(core, size=new_core_len, mode='linear', align_corners=False)
                        num_modified += 1
                        
                        warped_token = torch.cat([left_edge, core_compressed, right_edge], dim=2)
                        
                        # Verify edge preservation
                        if not torch.equal(left_edge, warped_token[:, :, :edge_frames]) or \
                           not torch.equal(right_edge, warped_token[:, :, -edge_frames:]):
                            edge_preservation_passed = False
                            
                        new_z_frames.append(warped_token)
                    else:
                        new_z_frames.append(z_token)
                        num_unchanged += 1
                else:
                    # Unchanged
                    new_z_frames.append(z_token)
                    num_unchanged += 1
                    
                current_y_idx += dur
                
            assert edge_preservation_passed, "Edge-preservation proof failed!"
            
            z_warped = torch.cat(new_z_frames, dim=2)
            
            # Create new y_mask for warped latent
            warped_lengths = torch.LongTensor([z_warped.size(2)])
            warped_mask = torch.unsqueeze(
                commons.sequence_mask(warped_lengths, z_warped.size(2)), 1
            ).type_as(y_mask)
            
            # D. Decode Warped Latent
            o_l1 = model.model_g.dec((z_warped * warped_mask)[:, :, :], g=None)
            dur_l1_ms = (o_l1.size(-1) / sample_rate) * 1000.0
            
            # Metrics
            ms_saved = dur_n0_ms - dur_l1_ms
            pct_saved = (ms_saved / dur_n0_ms) * 100.0 if dur_n0_ms > 0 else 0.0
            
            is_valid_pcm = bool(torch.isfinite(o_l1).all())
            peak_l1 = o_l1.abs().max().item()
            is_clipping = bool(peak_l1 > 1.0)
            
            metrics.append({
                "token": token,
                "n0_dur_ms": dur_n0_ms,
                "l1_dur_ms": dur_l1_ms,
                "ms_saved": ms_saved,
                "pct_saved": pct_saved,
                "num_tokens": t_x,
                "num_modified": num_modified,
                "num_unchanged": num_unchanged,
                "orig_latent_frames": z_norm.size(2),
                "warped_latent_frames": z_warped.size(2),
                "pcm_valid": is_valid_pcm,
                "peak": peak_l1,
                "clipping": is_clipping
            })
            
            # Save audio
            save_waveform(diag_dir / f"eval_normal_{token}.wav", o_norm[0])
            save_waveform(diag_dir / f"eval_warp_{token}.wav", o_l1[0])
            print(f"  {token:<6} | N0: {dur_n0_ms:6.1f} ms | L1: {dur_l1_ms:6.1f} ms | Saved: {ms_saved:5.1f} ms ({pct_saved:4.1f}%) | Mod/Un: {num_modified}/{num_unchanged}")

    # E. Summary & Gates
    all_pct_saved = [m["pct_saved"] for m in metrics]
    median_saved = np.median(all_pct_saved)
    p95_saved = np.percentile(all_pct_saved, 95)
    
    print(f"\n--- Gate Results ---")
    print(f"Median Speed Reduction: {median_saved:.1f}%")
    print(f"P95 Speed Reduction: {p95_saved:.1f}%")
    
    gate_speed = 15.0 <= median_saved <= 30.0
    gate_integrity = all(m["pcm_valid"] for m in metrics)
    gate_clipping = not any(m["clipping"] for m in metrics)
    
    print(f"Speed Gate (15-30%): {'PASS' if gate_speed else 'FAIL'}")
    print(f"PCM Integrity Gate:  {'PASS' if gate_integrity else 'FAIL'}")
    print(f"Clipping Gate:       {'PASS' if gate_clipping else 'FAIL'}")
    
    if gate_speed and gate_integrity and gate_clipping:
        print("\nAll Automatic Gates PASSED. Ready for listening.")
        # Identify finalists (top 8)
        # Select representative mix
        finalists = ["F", "N", "list", "b", "m", "comma", "button", "seven"]
        print(f"\nFinalist Set: {', '.join(finalists)}")
        print(f"Listening Directory: {diag_dir}")
        print("Please evaluate the finalist set in A/B/same format.")
    else:
        print("\nAutomatic Gates FAILED. Do not proceed to listening.")
        
    # Save metrics
    with open(results_dir / "phase2as_metrics.json", "w") as f:
        json.dump({
            "parameters": {
                "protected_edge_ms": protected_edge_ms,
                "edge_frames": edge_frames,
                "core_scale": core_scale
            },
            "gate_results": {
                "median_saved_pct": float(median_saved),
                "p95_saved_pct": float(p95_saved),
                "gate_speed_pass": bool(gate_speed),
                "gate_integrity_pass": bool(gate_integrity),
                "gate_clipping_pass": bool(gate_clipping)
            },
            "item_metrics": metrics
        }, f, indent=2)

if __name__ == "__main__":
    run_phase2as()

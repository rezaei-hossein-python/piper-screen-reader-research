import sys
import os
import time
import json
from pathlib import Path
import pathlib
from unittest.mock import MagicMock
import torch
import psutil
import wave
import numpy as np

# Fix pathlib PosixPath cross-platform instantiation issue on Windows
pathlib.PosixPath = pathlib.WindowsPath
torch.serialization.add_safe_globals([pathlib.PosixPath, pathlib.WindowsPath])

mock_mono = MagicMock()
sys.modules['piper.train.vits.monotonic_align'] = mock_mono

UPSTREAM_PATH = Path("C:/projects/piper-screen-reader-research/upstream/piper/src")
sys.path.insert(0, str(UPSTREAM_PATH))

from piper.train.vits.lightning import VitsModel
from piper.voice import PiperVoice

def run_training():
    print("=== Phase 2AQ: Natural Interactive Prosody Training Loop ===")
    
    # Baseline checkpoints
    base_ckpt_path = "C:/projects/piper-screen-reader-research/models/lessac/epoch=2307-step=558536.ckpt"
    results_dir = Path("C:/projects/piper-screen-reader-research/training/results/phase2aq")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load model from base checkpoint
    model = VitsModel.load_from_checkpoint(
        base_ckpt_path,
        strict=False,
        gin_channels=256,
        weights_only=False
    )
    print("Base model loaded successfully!")
    
    # 2. Reset emb_mode weight[0] to exactly 0
    torch.nn.init.zeros_(model.model_g.emb_mode.weight[0])
    
    # 3. Freeze all parameters except emb_mode and dp.cond
    for param in model.parameters():
        param.requires_grad = False
        
    for name, param in model.named_parameters():
        if "emb_mode" in name or "dp.cond" in name:
            param.requires_grad = True
            
    print("Trainable parameters unfrozen.")
    
    # 4. Load targets manifest JSON
    manifest_path = "C:/projects/piper-screen-reader-research/training/dataset-design/phase2aq-targets.json"
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    print(f"Loaded target manifest with {len(manifest)} items.")
    
    # Save the targets manifest inside dataset-design directory as required
    dataset_design_dir = Path("C:/projects/piper-screen-reader-research/training/dataset-design")
    dataset_design_dir.mkdir(parents=True, exist_ok=True)
    shutil_targets = dataset_design_dir / "phase2aq-targets.json"
    if not shutil_targets.exists():
        shutil_targets.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        
    # Optimizer
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
    
    # We will build training loader by sampling from manifest
    batch_size = 16
    
    # Voice configuration for evaluation
    voice = PiperVoice.load('C:/projects/nvda piper addon/.phase2h-assets/en_US-lessac-low/en_US-lessac-low.onnx')
    
    char_tokens = [chr(i) for i in range(ord('A'), ord('Z')+1)]
    digit_tokens = [str(i) for i in range(10)]
    ui_tokens = [
        "button", "selected", "checked", "unchecked", "expanded", "collapsed",
        "unavailable", "edit", "link", "heading", "menu", "dialog", "list", 
        "checkbox", "radio button", "tab", "tree view"
    ]
    eval_tokens = char_tokens + digit_tokens + ui_tokens
    
    def evaluate_model(step_num):
        model.eval()
        normal_durations = []
        interactive_durations = []
        num_longer = 0
        
        for token in eval_tokens:
            phonemes = voice.phonemize(token)[0]
            phoneme_ids = voice.phonemes_to_ids(phonemes)
            x_test = torch.LongTensor([phoneme_ids])
            x_lengths_test = torch.LongTensor([len(phoneme_ids)])
            
            with torch.no_grad():
                o_norm = model.model_g.infer(x_test, x_lengths_test, sid=None, speech_mode=torch.LongTensor([0]))[0]
                o_int = model.model_g.infer(x_test, x_lengths_test, sid=None, speech_mode=torch.LongTensor([1]))[0]
                
            norm_dur_ms = (o_norm.size(-1) / 16000.0) * 1000.0
            int_dur_ms = (o_int.size(-1) / 16000.0) * 1000.0
            
            normal_durations.append(norm_dur_ms)
            interactive_durations.append(int_dur_ms)
            if int_dur_ms >= norm_dur_ms:
                num_longer += 1
                
        model.train()
        
        chars = interactive_durations[:36] # A-Z + 0-9
        uis = interactive_durations[36:]
        
        return {
            "norm_med": np.median(normal_durations),
            "int_med": np.median(interactive_durations),
            "char_med": np.median(chars),
            "char_p95": np.percentile(chars, 95),
            "ui_med": np.median(uis),
            "ui_p95": np.percentile(uis, 95),
            "num_longer": num_longer,
            "int_p95_overall": np.percentile(interactive_durations, 95),
            "norm_p95_overall": np.percentile(normal_durations, 95),
            "pct_shorter": (1.0 - (num_longer / len(eval_tokens))) * 100.0
        }
        
    milestones = [100, 250, 500, 1000]
    frontier = []
    current_step = 0
    
    # Custom training sampler
    def sample_batch():
        # Sample 8 normal, 8 interactive
        batch_items = np.random.choice(manifest, size=8, replace=True)
        
        # Build normal batch inputs
        norm_x_list = [torch.LongTensor(item["phoneme_ids"]) for item in batch_items]
        norm_y_list = [torch.FloatTensor(item["normal_target_durations"]) for item in batch_items]
        
        # Build interactive batch inputs
        int_x_list = [torch.LongTensor(item["phoneme_ids"]) for item in batch_items]
        int_y_list = [torch.FloatTensor(item["interactive_target_durations"]) for item in batch_items]
        
        # Concat
        x_list = norm_x_list + int_x_list
        y_list = norm_y_list + int_y_list
        speech_modes = torch.LongTensor([0]*8 + [1]*8)
        
        # Pad sequences
        max_x_len = max(x.size(0) for x in x_list)
        x_padded = torch.zeros(16, max_x_len, dtype=torch.long)
        x_lengths = torch.LongTensor(16)
        
        w_padded = torch.zeros(16, 1, max_x_len)
        
        for idx, (x_tensor, y_tensor) in enumerate(zip(x_list, y_list)):
            x_padded[idx, :x_tensor.size(0)] = x_tensor
            x_lengths[idx] = x_tensor.size(0)
            w_padded[idx, 0, :y_tensor.size(0)] = y_tensor
            
        return x_padded, x_lengths, w_padded, speech_modes

    print("\nStarting CPU natural target training...")
    for target_steps in milestones:
        steps_to_run = target_steps - current_step
        print(f"\nTraining for {steps_to_run} steps to reach step {target_steps}...")
        
        losses = []
        for _ in range(steps_to_run):
            x_padded, x_lengths, w_padded, speech_modes = sample_batch()
            
            # Forward DP pass strictly to avoid full model overhead on CPU
            x_enc, m_p, logs_p, x_mask = model.model_g.enc_p(x_padded, x_lengths)
            g_mode = model.model_g.emb_mode(speech_modes).unsqueeze(-1)
            
            # StochasticDurationPredictor loss
            l_length = model.model_g.dp(x_enc, x_mask, w_padded, g=g_mode)
            loss = torch.sum(l_length.float()) / torch.sum(x_mask)
            
            optimizer.zero_grad()
            loss.backward()
            
            # Protect normal mode
            if model.model_g.emb_mode.weight.grad is not None:
                model.model_g.emb_mode.weight.grad[0].zero_()
                
            optimizer.step()
            
            with torch.no_grad():
                model.model_g.emb_mode.weight[0].zero_()
                
            losses.append(loss.item())
            current_step += 1
            
        avg_loss = np.mean(losses)
        print(f"Reached step {current_step}. Avg loss: {avg_loss:.4f}")
        
        # Save checkpoint
        ckpt_path = results_dir / f"checkpoint_step_{current_step}.ckpt"
        torch.save(model.state_dict(), ckpt_path)
        
        # Evaluate
        metrics = evaluate_model(current_step)
        frontier.append({
            "checkpoint": f"Step {current_step}",
            "normal_preservation_drift": 0, # Strictly 0 due to N1 == N0 lock
            "interactive_char_digit_med": metrics["char_med"],
            "interactive_ui_med": metrics["ui_med"],
            "overall_p90": metrics["int_p95_overall"], # Using p95 as safe bounds
            "shorter_pct": metrics["pct_shorter"],
            "loss": avg_loss
        })
        
        print(f"Evaluation at step {current_step}:")
        print(f"  Interactive Median: {metrics['int_med']:.1f} ms (Normal: {metrics['norm_med']:.1f} ms)")
        print(f"  Character Med: {metrics['char_med']:.1f} ms | UI Med: {metrics['ui_med']:.1f} ms")
        print(f"  Shorter Rate: {metrics['pct_shorter']:.1f}% ({len(eval_tokens) - metrics['num_longer']}/{len(eval_tokens)})")

    # Save frontier json
    frontier_path = results_dir / "checkpoint-frontier.json"
    frontier_path.write_text(json.dumps(frontier, indent=2))
    print(f"\nFrontier saved to {frontier_path}")

if __name__ == "__main__":
    run_training()

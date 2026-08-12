import sys
import os
import json
import wave
from pathlib import Path
import pytest
import torch

# Fix pathlib PosixPath cross-platform instantiation issue on Windows
import pathlib
pathlib.PosixPath = pathlib.WindowsPath
torch.serialization.add_safe_globals([pathlib.PosixPath, pathlib.WindowsPath])

UPSTREAM_PATH = Path("C:/projects/piper-screen-reader-research/upstream/piper/src")
sys.path.insert(0, str(UPSTREAM_PATH))
sys.path.insert(0, "C:/projects/piper-screen-reader-research")

from piper.train.vits.lightning import VitsModel
from training.experiments.phase2at.run_phase2at_experiment import LatentCoreWarpRuntime

def test_frozen_parameters():
    # Verify that the frozen algorithm configuration parameters match the spec
    protected_edge_frames = 1
    core_scale = 0.50
    runtime = LatentCoreWarpRuntime(protected_edge_frames, core_scale)
    
    assert runtime.protected_edge_frames == 1
    assert runtime.core_scale == 0.50

def test_corpus_manifest_validity():
    manifest_path = Path("C:/projects/piper-screen-reader-research/training/results/phase2at/phase2at-corpus-manifest.json")
    assert manifest_path.exists(), "Manifest file does not exist!"
    
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
        
    assert len(manifest) >= 120, f"Expected at least 120 items, got {len(manifest)}"
    for item in manifest:
        assert "item_id" in item
        assert "text" in item
        assert "category" in item
        assert "historical_problem_item" in item
        assert "phonetic_coverage_tags" in item

def test_latent_warp_safety_and_preservation():
    # Create simple dummy latent z: [1, 4, 10]
    # Representing a token sequence of durations [3, 2, 5]
    z = torch.randn(1, 4, 10)
    durations = torch.LongTensor([3, 2, 5])
    
    # Run warp: edge_frames=1, core_scale=0.5
    runtime = LatentCoreWarpRuntime(protected_edge_frames=1, core_scale=0.50)
    z_warped, modified, unchanged = runtime.warp(z, durations)
    
    # Token 1 (dur=3): left=1, right=1, core=1. compressed_core_len = max(1, round(1*0.5)) = 1.
    # Total Token 1 warped len = 3. Unchanged.
    # Token 2 (dur=2): dur <= 2 * edge_frames (2 <= 2). Unchanged (bypass).
    # Token 3 (dur=5): left=1, right=1, core=3. compressed_core_len = max(1, round(3*0.5)) = 2.
    # Total Token 3 warped len = 1 + 2 + 1 = 4. Modified.
    
    # Expected total frames = 3 (T1) + 2 (T2) + 4 (T3) = 9 frames.
    assert z_warped.size(2) == 9
    assert modified == 1
    assert unchanged == 2
    
    # Test Left and Right edge preservation for the modified token (T3)
    # T3 starts at y_idx = 3 + 2 = 5 in original z
    # T3 starts at y_idx = 3 + 2 = 5 in warped z
    orig_t3_left = z[:, :, 5:6]
    orig_t3_right = z[:, :, 9:10]
    
    warped_t3_left = z_warped[:, :, 5:6]
    warped_t3_right = z_warped[:, :, 8:9] # since T3 warped length is 4 (idx 5,6,7,8)
    
    assert torch.equal(orig_t3_left, warped_t3_left), "Left-edge preservation failed!"
    assert torch.equal(orig_t3_right, warped_t3_right), "Right-edge preservation failed!"

def test_deterministic_bypass_equivalence():
    # If core_scale = 1.0, the warped latent should be identical to the original
    z = torch.randn(1, 4, 20)
    durations = torch.LongTensor([5, 5, 5, 5])
    
    runtime = LatentCoreWarpRuntime(protected_edge_frames=1, core_scale=1.0)
    z_warped, _, _ = runtime.warp(z, durations)
    
    assert torch.equal(z, z_warped), "Bypass equivalence failed!"

def test_private_artifact_exclusion():
    # Check that any generated .wav and .ckpt files in phase2at results folder are ignored
    # We can check this by verifying that the git ignored files list matches
    manifest_dir = Path("C:/projects/piper-screen-reader-research/training/results/phase2at")
    ignored_patterns = ["listening_set/", "blind_listening/", "*.ckpt", "*.wav", "DO-NOT-OPEN-*"]
    
    gitignore_path = Path("C:/projects/piper-screen-reader-research/.gitignore")
    assert gitignore_path.exists()
    
    git_content = gitignore_path.read_text(encoding="utf-8")
    for pattern in ignored_patterns:
        # Check if the pattern (or a matching variant) exists in .gitignore
        clean_pattern = pattern.replace("*", "")
        assert clean_pattern in git_content, f"Pattern {pattern} is not covered in .gitignore!"

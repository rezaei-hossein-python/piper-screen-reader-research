import sys
import os
import random
import json
import shutil
from pathlib import Path

def generate_blind_test():
    results_dir = Path("C:/projects/piper-screen-reader-research/training/results/phase2as")
    listening_dir = results_dir / "listening_set"
    blind_dir = results_dir / "blind_listening"
    blind_dir.mkdir(parents=True, exist_ok=True)
    
    # 8 finalists
    finalists = ["F", "N", "list", "b", "m", "comma", "button", "seven"]
    
    answer_key = []
    
    random.seed(42) # Fixed seed for reproducibility of the key if needed, or better, no seed.
    # Let's use a dynamic seed but save the key carefully.
    import time
    random.seed(time.time())
    
    for i, token in enumerate(finalists):
        trial_num = i + 1
        
        n0_path = listening_dir / f"eval_normal_{token}.wav"
        l1_path = listening_dir / f"eval_warp_{token}.wav"
        
        # Randomize A and B
        options = [
            ("N0_Baseline", n0_path),
            ("L1_LatentWarp", l1_path)
        ]
        random.shuffle(options)
        
        a_id, a_src = options[0]
        b_id, b_src = options[1]
        
        a_dst = blind_dir / f"trial_{trial_num:02d}_A.wav"
        b_dst = blind_dir / f"trial_{trial_num:02d}_B.wav"
        
        shutil.copy2(a_src, a_dst)
        shutil.copy2(b_src, b_dst)
        
        answer_key.append({
            "trial": f"{trial_num:02d}",
            "token": token,
            "A": a_id,
            "B": b_id
        })
        
    key_path = results_dir / "DO-NOT-OPEN-phase2as-key.json"
    with open(key_path, "w") as f:
        json.dump(answer_key, f, indent=2)
        
    print(f"Blind listening set generated at: {blind_dir}")
    print(f"Answer key generated at: {key_path} (Do not open until scored)")
    
if __name__ == "__main__":
    generate_blind_test()

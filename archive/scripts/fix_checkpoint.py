#!/usr/bin/env python3
"""Fix operator checkpoint by removing _orig_mod. prefix from compiled model."""

import torch
import sys

def fix_checkpoint(input_path, output_path):
    print(f"Loading checkpoint from {input_path}")
    ckpt = torch.load(input_path, map_location="cpu")
    
    print(f"Checkpoint type: {type(ckpt)}")
    print(f"Checkpoint keys: {list(ckpt.keys()) if isinstance(ckpt, dict) else 'N/A'}")
    
    fixed = False
    
    # Case 1: Direct state dict with _orig_mod. prefix
    if isinstance(ckpt, dict) and any(k.startswith("_orig_mod.") for k in ckpt.keys()):
        print("Found _orig_mod. prefix in state dict keys")
        new_ckpt = {}
        for key, value in ckpt.items():
            new_key = key.replace("_orig_mod.", "")
            new_ckpt[new_key] = value
        
        torch.save(new_ckpt, output_path)
        print(f"✓ Fixed {len(new_ckpt)} parameters")
        print(f"Sample keys: {list(new_ckpt.keys())[:3]}")
        fixed = True
    
    # Case 2: Nested checkpoint with model key
    elif isinstance(ckpt, dict) and "model" in ckpt:
        model_state = ckpt["model"]
        if any(k.startswith("_orig_mod.") for k in model_state.keys()):
            print("Found _orig_mod. prefix in model state dict")
            new_model_state = {}
            for key, value in model_state.items():
                new_key = key.replace("_orig_mod.", "")
                new_model_state[new_key] = value
            
            ckpt["model"] = new_model_state
            torch.save(ckpt, output_path)
            print(f"✓ Fixed {len(new_model_state)} parameters")
            fixed = True
    
    if not fixed:
        print("❌ No _orig_mod. prefix found or unknown checkpoint structure")
        print(f"First 5 keys: {list(ckpt.keys())[:5] if isinstance(ckpt, dict) else 'N/A'}")
        return False
    
    return True

if __name__ == "__main__":
    input_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/operator.pt"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "checkpoints/operator_fixed.pt"
    
    success = fix_checkpoint(input_path, output_path)
    sys.exit(0 if success else 1)






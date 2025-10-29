#!/usr/bin/env python3
"""Simple config validation for ARM fixes (no inference, fast on Mac).

This script validates:
1. Config loads correctly
2. ARM parameters are set as expected
3. Debug flags are enabled
4. No syntax errors

Safe to run on M4 Mac - no model loading or GPU needed.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ups.utils.config_loader import load_config_with_includes


def validate_arm_config(config_path: str):
    """Validate ARM config without loading models."""

    print("=" * 70)
    print("ARM CONFIG VALIDATION (Local - No GPU Required)")
    print("=" * 70)

    # Load config
    print(f"\n1. Loading config: {config_path}")
    try:
        cfg = load_config_with_includes(config_path)
        print("   ✅ Config loaded successfully")
    except Exception as e:
        print(f"   ❌ FAILED to load config: {e}")
        return False

    # Check TTC section
    print("\n2. Checking TTC configuration...")
    ttc_cfg = cfg.get("ttc", {})

    if not ttc_cfg:
        print("   ❌ No TTC config found")
        return False

    if not ttc_cfg.get("enabled", False):
        print("   ⚠️  WARNING: TTC is disabled")
    else:
        print("   ✅ TTC enabled")

    # Check key fixes
    print("\n3. Validating ARM fixes...")

    fixes_status = []

    # Fix 1: Horizon > 1 for lookahead
    horizon = ttc_cfg.get("horizon", 1)
    if horizon > 1:
        print(f"   ✅ Lookahead enabled (horizon={horizon})")
        fixes_status.append(True)
    else:
        print(f"   ❌ Lookahead disabled (horizon={horizon}, should be >1)")
        fixes_status.append(False)

    # Fix 2: Conservation penalties disabled
    reward_cfg = ttc_cfg.get("reward", {})
    weights = reward_cfg.get("weights", {})

    mass_weight = weights.get("mass", 1.0)
    energy_weight = weights.get("energy", 0.0)
    neg_weight = weights.get("penalty_negative", 0.0)

    if mass_weight == 0.0 and energy_weight == 0.0:
        print(f"   ✅ Conservation penalties disabled (mass={mass_weight}, energy={energy_weight})")
        fixes_status.append(True)
    else:
        print(f"   ❌ Conservation penalties enabled (mass={mass_weight}, energy={energy_weight})")
        print(f"      For Burgers (dissipative PDE), these should be 0.0")
        fixes_status.append(False)

    if neg_weight > 0:
        print(f"   ✅ Negativity penalty enabled ({neg_weight})")
    else:
        print(f"   ⚠️  Negativity penalty disabled ({neg_weight})")

    # Fix 3: Increased diversity
    sampler_cfg = ttc_cfg.get("sampler", {})
    tau_range = sampler_cfg.get("tau_range", [0.15, 0.85])
    noise_std = sampler_cfg.get("noise_std", 0.05)

    tau_span = tau_range[1] - tau_range[0]
    if tau_span >= 0.8:
        print(f"   ✅ Wide tau range: {tau_range} (span={tau_span:.2f})")
        fixes_status.append(True)
    else:
        print(f"   ⚠️  Narrow tau range: {tau_range} (span={tau_span:.2f}, recommend >0.8)")
        fixes_status.append(False)

    if noise_std >= 0.08:
        print(f"   ✅ High noise: {noise_std}")
        fixes_status.append(True)
    else:
        print(f"   ⚠️  Low noise: {noise_std} (recommend >0.08)")
        fixes_status.append(False)

    # Fix 4: Debug enabled
    debug = ttc_cfg.get("debug", False) or reward_cfg.get("debug", False)
    if debug:
        print(f"   ✅ Debug logging enabled")
        fixes_status.append(True)
    else:
        print(f"   ⚠️  Debug logging disabled (won't see detailed ARM output)")
        fixes_status.append(False)

    # Summary
    print("\n4. Summary")
    print("   " + "=" * 60)

    fixes_applied = sum(fixes_status)
    total_fixes = len(fixes_status)

    print(f"   Fixes applied: {fixes_applied}/{total_fixes}")

    if fixes_applied == total_fixes:
        print("   🎉 ALL FIXES VALIDATED!")
        print("\n5. Next Steps:")
        print("   Run full evaluation on VastAI with:")
        print(f"   python scripts/vast_launch.py launch --config {config_path} --auto-shutdown")
        return True
    elif fixes_applied >= total_fixes - 1:
        print("   ✅ MOSTLY GOOD - Minor warnings only")
        print("\n5. Next Steps:")
        print("   Config is acceptable. Test on VastAI:")
        print(f"   python scripts/vast_launch.py launch --config {config_path} --auto-shutdown")
        return True
    else:
        print("   ❌ FIXES INCOMPLETE")
        print("\n5. Next Steps:")
        print("   Review config and ensure all fixes are applied")
        print(f"   See: configs/eval_burgers_arm_fixed.yaml for reference")
        return False


def main():
    config_path = "configs/eval_burgers_arm_fixed.yaml"

    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    success = validate_arm_config(config_path)

    print("\n" + "=" * 70)
    if success:
        print("✅ VALIDATION PASSED")
    else:
        print("❌ VALIDATION FAILED")
    print("=" * 70 + "\n")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

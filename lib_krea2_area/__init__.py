"""AReA (Adaptive Receptive-Field Attention) for Krea 2 in Forge Neo: shared constants.

Copyright 2026 Oleg Afonin. Licensed under the Apache License, Version 2.0; see LICENSE.

`schedule.py`, `tiling.py`, `bridge.py` and `attention.py` are vendored from
github.com/KooroshRH/AReA (commit 9e38ec2, Apache-2.0); see NOTICE for what changed.
"""

VERSION = "0.1.0"

LOG_PREFIX = "[Krea2 AReA]"
INFOTEXT_KEY = "Krea2 AReA"

#   native extent in patches per side (2x2 patches on the 8x latent, so 16 px per patch)
NATIVE_RAW = 64          # RAW and RAW + Turbo LoRA: trained up to 1024 px
NATIVE_TURBO = 128       # Turbo: 1K to 2K by its README
NATIVE_CHOICES = {"RAW (64)": NATIVE_RAW, "Turbo (128)": NATIVE_TURBO}
NATIVE_CUSTOM = "Custom"

#   defaults from the paper, except the phase switch: the paper's Qwen-Image value 0.86 (FLUX.1 0.89,
#   FLUX.2 0.94) lets Krea 2 duplicate objects when a step falls just under it (windows from sigma
#   0.852 at 20 steps), 0.80 and 0.83 do not, and 0.5 echoes every edge; see docs/validation.md
PHASE_SWITCH_SIGMA = 0.80
POOLING_DEFAULT = "subsample"
HALO_DEFAULT = 8
BRIDGE_RATIO_DEFAULT = 0.5

#   the Qwen3-VL tap stack Krea 2 conditioning carries, `[B, txtlen, 12, 2560]`; the same
#   identifying constants `neo-krea2-negpip` detects the DiT on
KREA2_TAP_LAYERS = 12
KREA2_TAP_DIM = 2560


def log(message: str) -> None:
    """Console line with the extension prefix; ASCII only, for legacy Windows code pages."""
    try:
        print(f"{LOG_PREFIX} {message}")
    except UnicodeEncodeError:
        print(f"{LOG_PREFIX} {message.encode('ascii', 'replace').decode('ascii')}")

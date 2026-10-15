"""The Krea 2 adapter: detection, grid and text length, and the per-call decision of what
each attention forward runs.

Host facts this rests on (Forge Neo `backend/nn/krea.py`, pinned by `tests/krea_replica.py`):

* `SingleStreamDiT.forward(x, timesteps, context, attention_mask=None, transformer_options={})`
  pads `x` to the 2x2 patch, so the image grid is `ceil(H/2) x ceil(W/2)`, and concatenates
  `cat(context, img)` with the text first; `txtlen` is `context.shape[1]` (the fusion keeps
  the token axis of the `[B, txtlen, 12, 2560]` tap stack). No mask is ever passed.
* Edit mode (`dynamic_args.ref_latents` non-empty) puts reference tokens between text and
  image; AReA stands down for those calls.
* `Attention.forward(x, freqs, mask, transformer_options)` holds `B H L D` q, k, v after
  `qknorm`, rotates with `ck.apply_rope`, expands the 12 key/value heads to 48 with
  `repeat_interleave`, and calls `attention_function(..., skip_reshape=True)`, which returns
  `B L (H D)`.
"""

from __future__ import annotations

from typing import Any, Callable

import torch

from lib_krea2_area import KREA2_TAP_DIM, KREA2_TAP_LAYERS
from lib_krea2_area.state import CallState, RunState


def is_krea2_dit(dit: Any) -> bool:
    """Whether this diffusion model really is the Krea 2 `SingleStreamDiT`.

    The predicate of `neo-krea2-negpip/lib_krea2_negpip/dit.py`, verbatim: duck-typed on
    the identifying constants and the attributes the hooks touch, nothing else.
    """
    try:
        txtlayers = int(getattr(dit, "txtlayers", 0))
        txtdim = int(getattr(dit, "txtdim", 0))
    except (TypeError, ValueError):
        return False

    if txtlayers != KREA2_TAP_LAYERS or txtdim != KREA2_TAP_DIM:
        return False

    return hasattr(dit, "blocks") and hasattr(dit, "txtmlp") and hasattr(getattr(dit, "txtfusion", None), "refiner_blocks")


def image_grid(x: torch.Tensor, patch: int) -> tuple[int, int]:
    """`(h_, w_)` of the padded latent, as `SingleStreamDiT.forward` computes it."""
    height, width = int(x.shape[-2]), int(x.shape[-1])
    return -(-height // patch), -(-width // patch)


def edit_mode() -> bool:
    """Whether this call carries Krea 2 Edit reference latents."""
    try:
        from backend.args import dynamic_args
    except ImportError:
        return False
    return bool(getattr(dynamic_args, "ref_latents", None))


# ================================================================================ #


def begin_call(dit: Any, state: RunState, x: Any, context: Any, options: Any) -> CallState:
    """Decide what this DiT call runs. Called by the forward wrapper before the blocks.

    Phase 1 of the port: every call runs the original attention; the grid and text length
    are recorded so the stand-down logic and its log lines are in place.
    """
    state.calls += 1
    cur = CallState()
    state.current = cur

    if not torch.is_tensor(x) or not torch.is_tensor(context) or context.ndim < 3:
        return cur

    if not state.active_pass:
        state.say_once("hires-off", "hires-fix pass with 'Apply to hires-fix pass' off: plain attention")
        return cur

    if edit_mode():
        state.say_once("edit", "Krea 2 Edit reference latents present: plain attention for these calls")
        return cur

    h, w = image_grid(x, int(getattr(dit, "patch", 2)))
    cur.h, cur.w, cur.txtlen = h, w, int(context.shape[1])

    native = state.config.native
    if max(h, w) <= native:
        state.say_once(("fits", h, w), f"{h}x{w} patch grid fits the native extent {native}: plain attention")
    return cur


def attention_forward(attn: Any, original: Callable, state: RunState, cur: CallState,
                      x: torch.Tensor, freqs: Any, mask: Any, transformer_options: Any) -> torch.Tensor:
    """The body of a patched `Attention.forward` once a phase is set for this call."""
    #   keyword past the required positional, for the reason in negpip's `_hook_attention`:
    #   a parameter interposed before `transformer_options` must take its default, not ours
    return original(x, freqs=freqs, mask=mask, transformer_options=transformer_options)

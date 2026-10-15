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

import importlib
from typing import Any, Callable, Tuple

import torch
import torch.nn.functional as F

from backend.quant_ops import ck

from lib_krea2_area import KREA2_TAP_DIM, KREA2_TAP_LAYERS, attention
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


def _host_attention() -> Callable:
    """The attention function Krea 2's own `Attention.forward` calls, looked up per call:
    `backend/nn/krea.py` binds it by name at import, and another extension may have rebound
    it there (negpip forces SDPA that way), so the module attribute is the one to follow."""
    global _krea_module
    if _krea_module is None:
        _krea_module = importlib.import_module("backend.nn.krea")
    return _krea_module.attention_function


_krea_module = None


def _rope(dit: Any, txtlen: int, rows: torch.Tensor, cols: torch.Tensor, batch: int) -> torch.Tensor:
    """`dit.pe_embedder` on synthetic ids: text at position zero, image tokens at `(0, row, col)`,
    repeated over the batch exactly as `SingleStreamDiT.forward` builds its own ids."""
    ids = torch.zeros(1, txtlen + rows.numel(), 3, device=rows.device, dtype=torch.float32)
    ids[0, txtlen:, 1] = rows.float()
    ids[0, txtlen:, 2] = cols.float()
    return dit.pe_embedder(ids.repeat(batch, 1, 1))


def coarse_ropes(dit: Any, state: RunState, h: int, w: int, f: int, txtlen: int, batch: int,
                 device: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    """The Phase I RoPE pair (upstream `FluxBackbone._coarse_ropes`). Pooled keys sit on the
    `(h/f) x (w/f)` grid; queries keep full resolution and take the coarse position of their
    block, so both are measured in the same units and no position exceeds the coarse grid."""
    key = (h, w, f, txtlen, batch, str(device))
    if key not in state.ropes:
        if len(state.ropes) > 4:
            state.ropes.clear()
        pos = torch.arange(h * w, device=device)
        coarse = torch.arange((h // f) * (w // f), device=device)
        rope_q = _rope(dit, txtlen, pos // w // f, pos % w // f, batch)
        rope_kv = _rope(dit, txtlen, coarse // (w // f), coarse % (w // f), batch)
        state.ropes[key] = (rope_q, rope_kv)
    return state.ropes[key]


# ================================================================================ #


def begin_call(dit: Any, state: RunState, x: Any, context: Any, options: Any) -> CallState:
    """Decide what this DiT call runs. Called by the forward wrapper before the blocks."""
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

    options = options if isinstance(options, dict) else {}
    sampling, sigmas = options.get("sampling_sigmas"), options.get("sigmas")
    if sampling is None or sigmas is None:
        state.say_once("sigmas", "no sigmas in transformer_options (not a k-diffusion sampler?): plain attention")
        return cur
    sigma = float(sigmas.flatten()[0]) if torch.is_tensor(sigmas) else float(sigmas)

    resolved = state.schedule_for(sampling, sigma, h, w, cur.txtlen)
    if resolved is None:
        return cur
    schedule, step = resolved
    phase, size = schedule.at(step)
    cur.step, cur.size = step, size

    if phase == "coarse":
        cur.data["rope_q"], cur.data["rope_kv"] = coarse_ropes(dit, state, h, w, size, cur.txtlen, int(x.shape[0]), x.device)
        cur.phase = phase
    #   Phase II is not wired yet: those steps run the original attention
    return cur


def _qkv(attn: Any, x: torch.Tensor):
    """`Attention.forward` up to the rotation: normalised, unrotated `B H L D` q (48 heads),
    k and v (12 key/value heads), and the output gate. The module's own `wq`, `wk`, `wv`,
    `gate` and `qknorm` run, so quantised (int8, fp8) layers stay on their own paths."""
    q, k, v, gate = attn.wq(x), attn.wk(x), attn.wv(x), attn.gate(x)
    b, length = int(x.shape[0]), int(x.shape[1])
    q = q.view(b, length, attn.heads, -1).transpose(1, 2)
    k = k.view(b, length, attn.kvheads, -1).transpose(1, 2)
    v = v.view(b, length, attn.kvheads, -1).transpose(1, 2)
    q, k = attn.qknorm(q, k)
    return q, k, v, gate


def _expand(attn: Any, k: torch.Tensor, v: torch.Tensor):
    if attn.kvheads != attn.heads:
        rep = attn.heads // attn.kvheads
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    return k, v


def _coarse(attn: Any, block: int, state: RunState, cur: CallState, x: torch.Tensor,
            transformer_options: Any) -> torch.Tensor:
    """Phase I: every full-resolution query against the text keys plus one key per `f x f`
    block. Pooling runs on the 12 key/value heads before the grouped-query expansion, and
    q and the pooled k are rotated separately, each with its own frequencies."""
    q, k, v, gate = _qkv(attn, x)
    cfg = state.config
    k, v = attention.pool_keys_values(k.transpose(1, 2), v.transpose(1, 2), cur.txtlen, cur.h, cur.w, cur.size,
                                      cfg.pooling, seed=state.seed, step=cur.step, block=block)
    k, v = k.transpose(1, 2), v.transpose(1, 2)
    q = ck.apply_rope1(q, cur.data["rope_q"])
    k = ck.apply_rope1(k, cur.data["rope_kv"])
    k, v = _expand(attn, k, v)
    out = _host_attention()(q, k, v, attn.heads, mask=None, skip_reshape=True, transformer_options=transformer_options)
    return attn.wo(out * F.sigmoid(gate))


def attention_forward(attn: Any, block: int, original: Callable, state: RunState, cur: CallState,
                      x: torch.Tensor, freqs: Any, mask: Any, transformer_options: Any) -> torch.Tensor:
    """The body of a patched `Attention.forward` once a phase is set for this call."""
    if not cur.checked:
        cur.checked = True
        expected = cur.txtlen + cur.h * cur.w
        if int(x.shape[1]) != expected:
            state.say_once(("length", int(x.shape[1]), expected),
                           f"sequence of {int(x.shape[1])} tokens, expected {cur.txtlen} text + {cur.h}x{cur.w} image: plain attention")
            cur.phase = None

    if cur.phase == "coarse":
        return _coarse(attn, block, state, cur, x, transformer_options)

    #   keyword past the required positional, for the reason in negpip's `_hook_attention`:
    #   a parameter interposed before `transformer_options` must take its default, not ours
    return original(x, freqs=freqs, mask=mask, transformer_options=transformer_options)

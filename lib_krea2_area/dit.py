"""Install and restore of AReA on a Krea 2 `SingleStreamDiT` (the negpip idiom).

Three hooks per installed run, all removed by `uninstall`:

1. a wrapper on `dit.forward` that decides, before the blocks run, what this call does
   (`krea2.begin_call`), drops that decision when the call returns, and uninstalls
   everything if the call raises, so a failed generation never leaves the model patched
   for the next one (a forward pre-hook cannot see the exception, hence a wrapper);
2. a forward hook on `dit.first`, which sees the image token embeddings once per call;
3. a wrapper on `dit.blocks[i].attn.forward` for every block (never the text-fusion blocks),
   which runs the original forward unless a phase is set for this call.

Every wrapper reads the run state off the module it is attached to, so a wrapper that some
other extension has since wrapped again, and that therefore cannot be removed, falls
through to the original forward once `uninstall` has deleted that attribute.
"""

from __future__ import annotations

from functools import wraps
from typing import Any, Optional

from lib_krea2_area import krea2, log
from lib_krea2_area.state import RunState

STATE_ATTR = "_krea2_area_state"
ORIGINAL_FORWARD = "_krea2_area_original_forward"
BLOCK_ATTR = "_krea2_area_block"
HOOKS_ATTR = "_krea2_area_hooks"
MARKER = "_krea2_area"


def _restore_forward(obj: Any, original: Any) -> None:
    """Put `forward` back the way it was, instance dict included.

    Assigning the captured bound method back would leave an entry in `vars(obj)` that was
    never there, which keeps a reference cycle alive and shadows the class attribute for
    anyone patching later.
    """
    if getattr(original, "__func__", None) is getattr(type(obj), "forward", None):
        vars(obj).pop("forward", None)
    else:
        obj.forward = original


def _unwrap(obj: Any) -> None:
    original = getattr(obj, ORIGINAL_FORWARD, None)
    if original is not None:
        if getattr(obj.forward, MARKER, False):
            _restore_forward(obj, original)
        delattr(obj, ORIGINAL_FORWARD)
    for name in (STATE_ATTR, BLOCK_ATTR):
        if hasattr(obj, name):
            delattr(obj, name)


def _attention_modules(dit: Any):
    for block in getattr(dit, "blocks", []) or []:
        attn = getattr(block, "attn", None)
        if attn is not None:
            yield attn


# ================================================================================ #


def _wrap_attention(attn: Any, block: int, state: RunState) -> None:
    setattr(attn, STATE_ATTR, state)
    setattr(attn, BLOCK_ATTR, block)
    if getattr(attn, ORIGINAL_FORWARD, None) is not None:
        return

    original = attn.forward

    @wraps(original)
    def area_attention_forward(x, freqs=None, mask=None, transformer_options={}):
        st = getattr(attn, STATE_ATTR, None)
        cur = st.current if st is not None else None
        if cur is None or cur.phase is None or mask is not None or freqs is None:
            return original(x, freqs=freqs, mask=mask, transformer_options=transformer_options)
        return krea2.attention_forward(attn, getattr(attn, BLOCK_ATTR, 0), original, st, cur, x, freqs, mask, transformer_options)

    setattr(area_attention_forward, MARKER, True)
    setattr(attn, ORIGINAL_FORWARD, original)
    attn.forward = area_attention_forward


def _wrap_dit(dit: Any, state: RunState) -> None:
    setattr(dit, STATE_ATTR, state)
    if getattr(dit, ORIGINAL_FORWARD, None) is not None:
        return

    original = dit.forward

    @wraps(original)
    def area_forward(x, timesteps, context, *args, **kwargs):
        st = getattr(dit, STATE_ATTR, None)
        if st is None:
            return original(x, timesteps, context, *args, **kwargs)

        options = kwargs.get("transformer_options", args[1] if len(args) > 1 else None)
        try:
            krea2.begin_call(dit, st, x, context, options)
            return original(x, timesteps, context, *args, **kwargs)
        except BaseException:
            uninstall(dit)
            log("the model call raised; AReA patches removed")
            raise
        finally:
            st.current = None

    setattr(area_forward, MARKER, True)
    setattr(dit, ORIGINAL_FORWARD, original)
    dit.forward = area_forward


def _after_first(module: Any, args: Any, output: Any) -> None:
    """Forward hook on `dit.first`: the image token embeddings `[B, h*w, C]`, once per call."""
    state = getattr(module, STATE_ATTR, None)
    cur = state.current if state is not None else None
    if cur is None or cur.phase != "fine" or cur.data.get("layout") is not None:
        return None
    if output.ndim != 3 or int(output.shape[1]) != cur.h * cur.w:
        return None
    krea2.fine_layout(state, cur, output)
    return None


# ================================================================================ #


def install(dit: Any, state: RunState) -> int:
    """Hook the DiT forward, `dit.first` and every block's attention. Returns the number of
    attention modules hooked. Idempotent: a second call only swaps the run state."""
    _wrap_dit(dit, state)

    first = getattr(dit, "first", None)
    if first is not None:
        setattr(first, STATE_ATTR, state)
        if not getattr(dit, HOOKS_ATTR, None):
            setattr(dit, HOOKS_ATTR, [first.register_forward_hook(_after_first)])

    hooked = 0
    for i, attn in enumerate(_attention_modules(dit)):
        _wrap_attention(attn, i, state)
        hooked += 1
    return hooked


def uninstall(dit: Any) -> None:
    """Undo everything `install` did. Safe to call twice, and on a DiT never installed."""
    if dit is None:
        return

    for handle in getattr(dit, HOOKS_ATTR, None) or []:
        handle.remove()
    if hasattr(dit, HOOKS_ATTR):
        delattr(dit, HOOKS_ATTR)

    first = getattr(dit, "first", None)
    if first is not None and hasattr(first, STATE_ATTR):
        delattr(first, STATE_ATTR)

    for attn in _attention_modules(dit):
        _unwrap(attn)
    _unwrap(dit)


def installed_state(dit: Any) -> Optional[RunState]:
    """The run state the DiT is installed with, or None if it is not (or no longer) patched."""
    if dit is None or getattr(dit, ORIGINAL_FORWARD, None) is None:
        return None
    return getattr(dit, STATE_ATTR, None)

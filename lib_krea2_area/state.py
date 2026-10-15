"""Per-run state, shared by the DiT forward wrapper, the `first` hook and the 28 patched
attention forwards.

One `RunState` lives for one installed run (a `process_batch` install to its teardown).
`current` is the per-call part: the DiT forward wrapper fills it before the blocks run, every
block reads it, and the wrapper drops it when the call returns, so nothing from one model
call can leak into the next.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from lib_krea2_area import (
    BRIDGE_RATIO_DEFAULT,
    HALO_DEFAULT,
    NATIVE_RAW,
    PHASE_SWITCH_SIGMA,
    POOLING_DEFAULT,
    log,
)


@dataclass(frozen=True)
class AreaConfig:
    """What the UI asked for. Frozen and hashable, so it can sit in the install signature;
    the seed is not part of it, since it changes every batch and needs no reinstall."""

    native: int = NATIVE_RAW             # native extent, patches per side
    switch_sigma: float = PHASE_SWITCH_SIGMA
    pooling: str = POOLING_DEFAULT
    halo: int = HALO_DEFAULT             # largest Halo width, patches
    bridge_ratio: float = BRIDGE_RATIO_DEFAULT
    w_min: int = 0                       # smallest window side; 0 means the native extent
    apply_hr: bool = True
    log: bool = True

    @property
    def window_floor(self) -> int:
        return int(self.w_min) if self.w_min and self.w_min > 0 else int(self.native)


@dataclass
class CallState:
    """What one DiT call runs. `phase` None means the original attention for this call."""

    phase: Optional[str] = None          # None, "coarse" (Phase I) or "fine" (Phase II)
    step: int = 0
    size: int = 0                        # pooling factor f (Phase I) or window side w (Phase II)
    h: int = 0                           # image grid in patches, after padding
    w: int = 0
    txtlen: int = 0
    checked: bool = False                # the first block has cross-checked the sequence length
    data: Dict[str, Any] = field(default_factory=dict)


class RunState:
    def __init__(self, config: AreaConfig, seed: int = 0):
        self.config = config
        self.seed = int(seed)
        self.active_pass = True          # False while a hires pass runs with "Apply to hires-fix pass" off
        self.current: Optional[CallState] = None
        self.calls = 0                   # model calls seen since install, for the logs and the tests
        self._said: set = set()

    def say_once(self, key, message: str) -> None:
        """Log `message` the first time `key` comes up in this run, if logging is on."""
        if key in self._said:
            return
        self._said.add(key)
        if self.config.log:
            log(message)

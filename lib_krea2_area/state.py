"""Per-run state, shared by the DiT forward wrapper, the `first` hook and the 28 patched
attention forwards.

One `RunState` lives for one installed run (a `process_batch` install to its teardown).
`current` is the per-call part: the DiT forward wrapper fills it before the blocks run, every
block reads it, and the wrapper drops it when the call returns, so nothing from one model
call can leak into the next.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import torch

from lib_krea2_area import (
    BRIDGE_RATIO_DEFAULT,
    HALO_DEFAULT,
    NATIVE_RAW,
    PHASE_SWITCH_SIGMA,
    POOLING_DEFAULT,
    log,
)
from lib_krea2_area.schedule import F_MIN, Schedule, build_schedule, start_window, window_sizes, window_step_cost

PX_PER_PATCH = 16   # 8x VAE, 2x2 patches


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


def step_sigmas(sampling_sigmas: Any) -> Tuple[float, ...]:
    """The sigma each step evaluates the model at: the schedule minus its terminal entry."""
    values = sampling_sigmas.tolist() if torch.is_tensor(sampling_sigmas) else list(sampling_sigmas)
    values = [float(v) for v in values]
    return tuple(values[:-1]) if len(values) > 1 else tuple(values)


def nearest_step(sigmas: Sequence[float], sigma: float) -> int:
    """Index of the entry of `sigmas` nearest to `sigma`, ties to the earlier step. Nearest,
    not exact, so a sampler that evaluates between steps (Heun, DPM++ 2S) or a sigma patched
    at a SPEED transition still lands on the step it belongs to."""
    return min(range(len(sigmas)), key=lambda i: (abs(sigmas[i] - sigma), i))


def phase1_count(run: Sequence[float], switch: float) -> int:
    """Leading steps at or above the switch sigma: Phase I. The rest are Phase II."""
    count = 0
    for sigma in run:
        if sigma < switch:
            break
        count += 1
    return count


def bootstrap_schedule(h: int, w: int, steps: int, floor: int, txtlen: int) -> Schedule:
    """Phase II alone, for a pass that starts below the switch (img2img, hires fix).

    `build_schedule` with no Phase I hands the ladder over from the full canvas, so its first
    rung is one window over everything: full attention at an out-of-distribution size.
    Upstream's AReA-Bootstrap never builds that: it keeps the Phase I steps in the count
    (`step_offset`) and so starts its windows from the handoff of the last pooling factor,
    `F_MIN`. This does the same with the vendored pieces.
    """
    extent = max(h, w)
    floor = min(int(floor), extent)
    lo = max(floor, extent // F_MIN)
    windows = window_sizes(h, w, steps, start_window(h, w, lo, txtlen), floor, txtlen) if steps > 0 else []
    return Schedule((h, w), (), tuple(windows))


def attention_cost(schedule: Schedule, txtlen: int) -> float:
    """Mean attention cost of a step in units of one full-attention pass, windows alone (the
    Halo and bridge add to Phase II steps, at most up to one full pass each)."""
    h, w = schedule.grid
    n, t = float(h * w), float(txtlen)
    costs = [(t + n / (f * f)) / (t + n) for f in schedule.factors]
    costs += [window_step_cost(h, w, size, txtlen) for size in schedule.windows]
    return sum(costs) / max(1, len(costs))


class RunState:
    def __init__(self, config: AreaConfig, seed: int = 0):
        self.config = config
        self.seed = int(seed)
        self.active_pass = True          # False while a hires pass runs with "Apply to hires-fix pass" off
        self.current: Optional[CallState] = None
        self.calls = 0                   # model calls seen since install, for the logs and the tests
        self.ropes: Dict[Any, Any] = {}  # coarse RoPE pairs, keyed on (h, w, f, txtlen, batch, device)
        self._said: set = set()
        self._schedule_key = None
        self._schedule: Optional[Schedule] = None
        self._offset = 0

    def schedule_for(self, sampling_sigmas: Any, sigma: float, h: int, w: int,
                     txtlen: int) -> Optional[Tuple[Schedule, int]]:
        """`(schedule, step)` for this call, or None for plain attention.

        The schedule is built on the first call of a pass and kept while `(sampling_sigmas,
        h, w, txtlen)` stays the same. Its steps are the ones the pass actually runs: Forge
        hands img2img (and so the hires-fix pass) the whole schedule in `sampling_sigmas`
        but starts sampling partway down it, so the pass starts at the step nearest the
        first sigma seen. Phase I is the leading run of steps at or above the switch sigma;
        a pass that starts below it has no Phase I and a full Phase II ladder.
        """
        cfg = self.config
        if max(h, w) <= cfg.native:
            return None
        sigmas = step_sigmas(sampling_sigmas)
        if not sigmas:
            return None
        index = nearest_step(sigmas, sigma)

        key = (sigmas, h, w, txtlen)
        if key != self._schedule_key:
            self._schedule_key = key
            self._offset = index
            run = sigmas[index:]
            phase1 = phase1_count(run, cfg.switch_sigma)
            try:
                if phase1:
                    self._schedule = build_schedule(h, w, len(run), phase1, cfg.native, w_min=cfg.window_floor, txt_len=txtlen)
                else:
                    self._schedule = bootstrap_schedule(h, w, len(run), cfg.window_floor, txtlen)
            except ValueError as error:
                self._schedule = None
                self.say_once(("grid", h, w), f"{h}x{w} patch grid: {error}; plain attention")
            else:
                self._describe(self._schedule, run, txtlen)

        if self._schedule is None:
            return None
        step = min(max(0, index - self._offset), self._schedule.steps - 1)
        return self._schedule, step

    def _describe(self, schedule: Schedule, run: Sequence[float], txtlen: int) -> None:
        if not self.config.log:
            return
        h, w = schedule.grid
        log(f"{w * PX_PER_PATCH}x{h * PX_PER_PATCH} px ({h}x{w} patches), {txtlen} text tokens, native {self.config.native}: "
            f"{len(run)} steps from sigma {run[0]:.3f}, switch at {self.config.switch_sigma:g}; "
            f"Phase I {len(schedule.factors)} steps f={list(schedule.factors)}; "
            f"Phase II {len(schedule.windows)} steps w={list(schedule.windows)}; "
            f"attention ~{attention_cost(schedule, txtlen):.2f} of full before Halo and bridge")

    def say_once(self, key, message: str) -> None:
        """Log `message` the first time `key` comes up in this run, if logging is on."""
        if key in self._said:
            return
        self._said.add(key)
        if self.config.log:
            log(message)

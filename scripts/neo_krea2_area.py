"""
AReA (Adaptive Receptive-Field Attention) for Krea 2, a Forge Neo extension.

Copyright 2026 Oleg Afonin. Licensed under the Apache License, Version 2.0; see LICENSE.
Derived from the AReA sample code (github.com/KooroshRH/AReA, commit 9e38ec2, Apache-2.0),
by Javad Rajabi, Koorosh Roohi, David B. Lindell and Babak Taati; see NOTICE.

Beyond the model's native extent, Phase I pools the keys and values every full-resolution
query sees and Phase II restricts each query to a shrinking window with a Halo and a global
bridge. At or below the native extent the original attention runs, bit for bit.

Lifecycle (the negpip idiom): `process_batch` resolves the settings and installs on the
DiT, `postprocess` tears down, and a signature of `(id(model), id(dit), config)` keeps the
patches across the batches of one run. The DiT forward wrapper tears down by itself if a
model call raises, and `process_batch` reinstalls when it finds the patches gone.
"""

from __future__ import annotations

import gradio as gr

from lib_krea2_area import (
    BRIDGE_RATIO_DEFAULT,
    HALO_DEFAULT,
    INFOTEXT_KEY,
    NATIVE_CHOICES,
    NATIVE_CUSTOM,
    NATIVE_RAW,
    PHASE_SWITCH_SIGMA,
    POOLING_DEFAULT,
    log,
)
from lib_krea2_area.attention import POOLING
from lib_krea2_area.dit import install, installed_state, uninstall
from lib_krea2_area.krea2 import is_krea2_dit
from lib_krea2_area.state import AreaConfig, RunState

from modules import errors, scripts
from modules.ui_components import InputAccordion

#   `type(shared.sd_model).__name__` for `backend/diffusion_engine/krea.py`
KREA2_ENGINE = "Krea2"


def resolve_native(choice, custom) -> int:
    """The native extent in patches from the radio and the custom number."""
    if choice in NATIVE_CHOICES:
        return NATIVE_CHOICES[choice]
    try:
        return max(1, int(custom))
    except (TypeError, ValueError):
        return NATIVE_RAW


def format_infotext(config: AreaConfig, seed: int) -> str:
    text = (f"native={config.native} switch={config.switch_sigma:g} pool={config.pooling} "
            f"halo={config.halo} bridge={config.bridge_ratio:g}")
    if config.w_min:
        text += f" wmin={config.w_min}"
    if not config.apply_hr:
        text += " hr=off"
    return text + f" seed={seed}"


def parse_infotext(text) -> dict:
    """`key=value` pairs of the infotext entry; tolerant of anything else in the string."""
    out = {}
    for part in str(text or "").split():
        key, sep, value = part.partition("=")
        if sep:
            out[key] = value
    return out


class Krea2AReA(scripts.Script):
    sorting_priority = 16

    #   class attributes: `postprocess` is not guaranteed to see the same instance that
    #   `process_batch` ran on, and the patches outlive both
    state: RunState | None = None
    dit = None
    signature: tuple | None = None
    said: set = set()

    def title(self):
        return "AReA (Krea 2)"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with InputAccordion(False, label=self.title(), elem_id=self.elem_id("enable")) as enable:
            gr.Markdown("Coarse-to-fine attention for canvases beyond the model's native extent (AReA, arXiv:2610.17227). At or below the native extent the original attention runs.")

            with gr.Row():
                native = gr.Radio(
                    choices=[*NATIVE_CHOICES, NATIVE_CUSTOM],
                    value="RAW (64)",
                    label="Native extent",
                    info="patches per side the checkpoint was trained at (64 is 1024 px); AReA engages from 1.5x this and is a no-op below",
                )
                native_custom = gr.Number(value=NATIVE_RAW, precision=0, label="Custom native extent", info="patches per side, used when Native extent is Custom")

            switch = gr.Slider(
                minimum=0.5,
                maximum=1.0,
                step=0.005,
                value=PHASE_SWITCH_SIGMA,
                label="Phase switch sigma",
                info="Phase I (pooled keys) runs while sigma is at or above this, Phase II (windows) below; on Krea 2 0.80 to 0.83 holds the layout, 0.86 can duplicate objects (windows start too early), 0.5 echoes every edge (pooling runs too late)",
            )

            with gr.Row():
                pooling = gr.Radio(choices=list(POOLING), value=POOLING_DEFAULT, label="Pooling", info="how Phase I keeps one key per block")
                halo = gr.Slider(minimum=0, maximum=16, step=1, value=HALO_DEFAULT, label="Halo", info="largest ring width around a window, in patches")
                bridge = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=BRIDGE_RATIO_DEFAULT, label="Bridge ratio", info="(ratio x window side)^2 global tokens per window; 0 disables")

            with gr.Row():
                w_min = gr.Number(value=0, precision=0, label="Minimum window", info="smallest window side in patches; 0 is the native extent (calibration only)")
                seed = gr.Number(value=-1, precision=0, label="Seed", info="jitter, Halo widths and random pooling; -1 follows the generation seed")

            with gr.Row():
                apply_hr = gr.Checkbox(True, label="Apply to hires-fix pass", visible=not is_img2img)
                show_log = gr.Checkbox(True, label="Log", info="per pass: grid, phases, window ladder, estimated attention cost")

        def field(key, convert=str, default=None):
            def read(d):
                if INFOTEXT_KEY not in d:
                    return None
                value = parse_infotext(d[INFOTEXT_KEY]).get(key)
                if value is None:
                    return default
                try:
                    return convert(value)
                except ValueError:
                    return None
            return read

        def native_choice(d):
            value = field("native", int)(d)
            if value is None:
                return None
            for name, patches in NATIVE_CHOICES.items():
                if patches == value:
                    return name
            return NATIVE_CUSTOM

        self.infotext_fields = [
            (enable, lambda d: INFOTEXT_KEY in d),
            (native, native_choice),
            (native_custom, field("native", int)),
            (switch, field("switch", float)),
            (pooling, field("pool")),
            (halo, field("halo", int)),
            (bridge, field("bridge", float)),
            (w_min, field("wmin", int, 0)),
            (seed, field("seed", int)),
            (apply_hr, field("hr", lambda v: v != "off", True)),
        ]

        return [enable, native, native_custom, switch, pooling, halo, bridge, w_min, seed, apply_hr, show_log]

    # ============================================================================ #

    @classmethod
    def _teardown(cls):
        if cls.dit is not None:
            uninstall(cls.dit)
        cls.dit = None
        cls.state = None
        cls.signature = None

    @classmethod
    def _say_once(cls, key, message):
        if key not in cls.said:
            cls.said.add(key)
            log(message)

    def process_batch(self, p, enable, native, native_custom, switch, pooling, halo, bridge, w_min, seed, apply_hr, show_log, *args, **kwargs):
        cls = Krea2AReA

        if not enable:
            cls._teardown()
            return

        model = getattr(p, "sd_model", None)
        if model is None or type(model).__name__ != KREA2_ENGINE:
            cls._teardown()
            cls._say_once(("model", type(model).__name__), f"model is {type(model).__name__}, not Krea 2: AReA stands down")
            return

        try:
            dit = model.forge_objects.unet.model.diffusion_model
        except AttributeError:
            dit = None
        if not is_krea2_dit(dit):
            cls._teardown()
            cls._say_once("dit", "the diffusion model does not look like Krea 2's SingleStreamDiT: AReA stands down")
            return

        config = AreaConfig(
            native=resolve_native(native, native_custom),
            switch_sigma=min(1.0, max(0.0, float(switch))),
            pooling=pooling if pooling in POOLING else POOLING_DEFAULT,
            halo=max(0, int(halo)),
            bridge_ratio=max(0.0, float(bridge)),
            w_min=max(0, int(w_min or 0)),
            apply_hr=bool(apply_hr),
            log=bool(show_log),
        )

        resolved_seed = int(seed) if seed is not None else -1
        if resolved_seed < 0:
            batch_seeds = kwargs.get("seeds") or [getattr(p, "seed", 0)]
            resolved_seed = int(batch_seeds[0])

        signature = (id(model), id(dit), config)
        if cls.signature == signature and installed_state(dit) is cls.state and cls.state is not None:
            #   same model, same settings: keep the patches, take this batch's seed
            cls.state.seed = resolved_seed
        else:
            cls._teardown()
            state = RunState(config, resolved_seed)
            try:
                hooked = install(dit, state)
            except Exception:
                uninstall(dit)
                errors.report("Krea2 AReA: install failed, plain attention for this run", exc_info=True)
                return
            cls.dit, cls.state, cls.signature = dit, state, signature
            if config.log:
                log(f"installed on {hooked} attention blocks: {format_infotext(config, resolved_seed)}")

        p.extra_generation_params[INFOTEXT_KEY] = format_infotext(config, resolved_seed)

    def process_before_every_sampling(self, p, *args, **kwargs):
        state = Krea2AReA.state
        if state is not None:
            state.active_pass = not (getattr(p, "is_hr_pass", False) and not state.config.apply_hr)

    def postprocess(self, p, processed, *args):
        Krea2AReA._teardown()

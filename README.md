# Neo-Krea2-AReA

A Forge Neo extension that applies AReA, Adaptive Receptive-Field Attention (Rajabi, Roohi, Lindell and Taati, [arXiv:2610.17227](https://arxiv.org/abs/2610.17227)), to Krea 2 when the canvas is larger than the model's native extent. It is training-free: the latent is never resized, the sampler is never touched, and canvases at or below the native extent run the original attention.

Status: work in progress. Both phases are implemented: Phase I (pooled keys and values) above the switch sigma and Phase II (windows with Halo and bridge) below it, beyond the native extent. Validation and calibration on Krea 2 are under way, and defaults may still change. The full README (install, knobs, stand-down rules, composition with SPEED and hires fix, calibration table) follows once the defaults are measured.

## How it works, in two lines

Phase I pools the keys and values that every full-resolution query sees, on a grid that starts at the native size and ramps to twice native, so the layout forms at a scale the model knows. Phase II restricts each query to a window that shrinks step by step to the native side, extended by a Halo ring of neighbours and a few global bridge tokens, so detail is refined at the distances and key counts the model was trained on.

The switch between the phases is a sigma, not a step count, so the same setting holds across step budgets, and a hires-fix pass that starts below the switch runs Phase II only (the paper's AReA-Bootstrap).

## Credits and license

This extension is derived from the AReA sample code, [github.com/KooroshRH/AReA](https://github.com/KooroshRH/AReA) (commit 9e38ec2), by Javad Rajabi, Koorosh Roohi, David B. Lindell and Babak Taati, released under the [Apache License 2.0](https://github.com/KooroshRH/AReA/blob/main/LICENSE). Its four model-independent modules (`schedule.py`, `tiling.py`, `bridge.py` and `attention.py`) are vendored in `lib_krea2_area/` with a header line naming the source; `NOTICE` lists them and what was changed in each. The Krea 2 adapter follows the structure of upstream's FLUX.1 backbone adapter. All credit for the method belongs to the authors; any defect in the Krea 2 port is this project's.

If you use this work, please cite the paper:

```bibtex
@misc{rajabi2026area,
      title={AReA: Adaptive Receptive-Field Attention for Efficient High-Resolution Diffusion Transformers},
      author={Javad Rajabi and Koorosh Roohi and David B. Lindell and Babak Taati},
      year={2026},
      eprint={2610.17227},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2610.17227},
}
```

This extension is licensed under the Apache License 2.0, the same license as upstream; see `LICENSE` and `NOTICE`. Copyright 2026 Oleg Afonin.

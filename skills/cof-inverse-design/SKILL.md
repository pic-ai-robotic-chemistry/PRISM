---
name: cof-inverse-design
description: Use a pretrained CNN and Bayesian optimization to find red-phase parameters for a target CD peak wavelength or signed peak intensity.
---

# CPL Inverse Design

Find red-phase parameters for a target CD peak wavelength or signed peak intensity. The scripts use the bundled `assets/best.ckpt` while searching thickness and stretch.

## Usage

Run from the skill directory. Supply a target wavelength in nm with `-t`, a target peak intensity in mdeg with `--target_intensity`, or both.

```bash
python -m pip install "torch>=2.0" "numpy>=1.24" "scikit-optimize>=0.10" "matplotlib>=3.7"
python scripts/inverse_design.py --color B -t 550 --target_intensity 5000 --output-dir outputs/single
python scripts/run_inverse_design_batch.py --workers 2 --output-root outputs/batch
```

The batch command searches R, G, and B at 475, 525, 575, and 625 nm, each with a target peak intensity of +5000 mdeg. The `--inverse-design` option in [cpl-message-encoding](../cpl-message-encoding/SKILL.md) calls this skill.

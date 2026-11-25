---
name: cpl-predict-skill
description: Prepare JASCO CPL-300 exports, train a CNN to predict CD spectra, and evaluate checkpoints or cross-validation folds.
---

# CPL Spectrum Prediction

Predict and evaluate 801-point CD spectra from color, gray, thickness, stretch, and phase.

## Usage

Run from the skill directory. Name the JASCO exports `{color}-{gray}-{thickness}-{stretch}.txt` and put red-phase files in a `red phase` subdirectory. Use `--red-phase-dir DIR` to specify a different name. The parser reads CD from the fourth column of the `XYDATA` section.

```bash
python -m pip install 'torch>=2.0' 'pytorch-lightning>=2.0' 'hydra-core>=1.3' 'numpy>=1.24' 'matplotlib>=3.7'
python scripts/process_cpl_data.py --data-dir measurements --output-dir data
python scripts/train.py
python scripts/infer.py -c logs/training-run -d data/cpl_cd.pkl -k data/val_keys.txt
```

Replace `logs/training-run` with the run directory containing `checkpoints/best.ckpt`. Use the same CD dataset for training and evaluation. See the [cross-validation commands](references/workflow.md) for fold-based training and evaluation.

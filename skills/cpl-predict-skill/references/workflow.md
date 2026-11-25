# CPL Cross-validation

Generate train, validation, and held-out test keys from the CD pickle.

```bash
python scripts/create_cv_splits.py --data_path data/cpl_cd.pkl --out_dir cv_splits/seed_42 --n_folds 5
```

Train one model per fold and evaluate it on its validation keys and the shared test keys. The example is for fold 1. Replace `logs/fold1-run` with its actual training run directory.

```bash
python scripts/train.py tag=cpl_cv_fold1 train_keys_path=cv_splits/seed_42/fold_1/train_keys.txt val_keys_path=cv_splits/seed_42/fold_1/val_keys.txt
python scripts/infer.py -c logs/fold1-run -d data/cpl_cd.pkl -k cv_splits/seed_42/fold_1/val_keys.txt --output_tag val
python scripts/infer.py -c logs/fold1-run -d data/cpl_cd.pkl -k cv_splits/seed_42/test_keys.txt --output_tag test
```

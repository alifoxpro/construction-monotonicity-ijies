# Synthetic generator

Settings are the `config` object at the top of each results file (K, T, n_train, n_test, noise, strength, rework_len, lr, steps, reg_weight), and `rows` holds every per-seed outcome, including the measured ground-truth violation rate of each configuration. The generator itself is scripts/rework_sweep.py and scripts/rework_concentration.py; emissions are synthetic, so no images are involved and both files regenerate from the recorded settings.

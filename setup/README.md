# Setup guide

## Requirements

- **Python** 3.10 or newer (3.13 verified)
- **pip** recent enough to install wheels for your platform

## 1. Create a virtual environment

From the project root (`deadline_dccc_project/`):

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

## 2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install pytest             # tests are optional
```

### PyTorch (CUDA only)

`requirements.txt` pulls **CPU PyTorch** from PyPI which is sufficient for
this project — DRL training and evaluation both run on CPU in a few
minutes. If you specifically need a CUDA build, install it from the
[official install matrix](https://pytorch.org/get-started/locally/)
first, then install the rest:

```bash
pip install pandas matplotlib numpy
```

## 3. Verify the environment

```bash
python -m pytest tests/        # 7 smoke tests, < 1 s
python test_drl.py             # legacy DQN smoke check
```

## 4. Run the experiments

Full reproduction (5 seeds, all 7 policies, 50 DRL episodes, ≈ 6 min on
a 2024 MacBook Pro):

```bash
python main.py
```

Quick smoke run (under a minute, no DRL training):

```bash
python main.py --no-drl --seeds 42 --policies LRU LFU DCCC_HASH DEADLINE_HEURISTIC
```

Outputs land in `results/*.csv` (`per_seed`, `summary`,
`significance`, optional `drl_training_history`) and `plots/*.png`.

All knobs live in `configs/simulation_config.json` and can be
overridden on the CLI with `--seeds`, `--policies`, `--episodes`,
`--config`, `--output-dir`.

## Checkpoints

`models_saved/` holds per-seed DQN checkpoints written by
`drl.trainer.save_agents` (call from a notebook if you want to
checkpoint — `runner.py` retrains per seed by default and does not
persist). The directory is git-ignored.

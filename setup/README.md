# Setup guide

## Requirements

- **Python** 3.9 or newer (3.10+ recommended)
- **pip** current enough to install wheels for your platform

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
```

### PyTorch

`requirements.txt` pulls **PyTorch** from PyPI. For a **CPU-only** machine this is usually enough. If you need a specific CUDA build, install PyTorch from the [official install matrix](https://pytorch.org/get-started/locally/) first, then install the rest:

```bash
pip install pandas matplotlib
```

## 3. Verify the environment

```bash
python test_drl.py
```

You should see a selected action, training loss, and `DRL test completed`.

## 4. Run the full simulation

```bash
python main.py
```

This reads `configs/simulation_config.json`, runs all policies (LRU, LFU, DCCC_HASH, DEADLINE_DCCC), and writes:

- `results/*.csv` and per-policy logs (as configured)
- `plots/*.png` (matplotlib figures)

Edit `configs/simulation_config.json` to change request counts, cache size, DRL hyperparameters, etc.

## Checkpoints

The `models_saved/` directory may contain `*.pt` weight files from training runs. Those files are **not** committed to Git (see `.gitignore`); regenerate them locally if your workflow saves agents there.

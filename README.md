# Optimised Transformer-Based RUL Prediction via Knowledge Distillation

Capstone project: a high-capacity Transformer **teacher** is trained on
NASA C-MAPSS turbofan degradation data, then a lightweight **student**
model is trained via **knowledge distillation** (soft-target + feature
alignment) to approximate the teacher's accuracy at a fraction of the
compute cost. An LSTM baseline is included for reference.

## 1. File structure

```
rul_kd_project/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/                     # <- put the downloaded C-MAPSS .txt files here
│   │   ├── train_FD001.txt ... train_FD004.txt
│   │   ├── test_FD001.txt  ... test_FD004.txt
│   │   └── RUL_FD001.txt   ... RUL_FD004.txt
│   └── processed/               # auto-generated .npz windowed arrays
├── src/
│   ├── config.py                # ALL hyperparameters live here
│   ├── data/
│   │   ├── preprocessing.py     # load, normalise, RUL labels, sliding windows
│   │   └── dataset.py           # PyTorch Dataset / DataLoader
│   ├── models/
│   │   ├── layers.py            # positional encoding, regression head
│   │   ├── teacher_transformer.py
│   │   ├── student_model.py     # lightweight Transformer (+ optional GRU head)
│   │   └── lstm_baseline.py     # plain LSTM reference model
│   ├── distillation/
│   │   └── kd_loss.py           # task + soft-target + feature-alignment loss
│   ├── utils/
│   │   ├── metrics.py           # RMSE, NASA scoring function
│   │   └── complexity.py        # param count, FLOPs, inference time
│   ├── train_teacher.py         # Sec. 7.2
│   ├── train_student_kd.py      # Sec. 7.3 + 7.4
│   ├── evaluate.py              # Sec. 7.5 — final comparison table
│   └── trade_off_sweep.py       # Sec. 7.6 — compression sweep
├── scripts/
│   └── run_pipeline.sh          # runs steps 1-4 end to end for one subset
├── checkpoints/                 # saved model weights (.pt)
├── results/                     # output CSVs (comparison tables, sweeps)
└── notebooks/                   # optional exploratory / plotting notebooks
```

## 2. Environment setup

Requires Python 3.10 or 3.11.

```bash
# from the rul_kd_project/ directory
python3 -m venv .venv

# activate it
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows (PowerShell: .venv\Scripts\Activate.ps1)

# upgrade pip, then install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

If you have an NVIDIA GPU, replace the plain `torch` line in
`requirements.txt` with the CUDA-matched install command from
https://pytorch.org/get-started/locally/ (e.g. for CUDA 12.1:
`pip install torch --index-url https://download.pytorch.org/whl/cu121`)
**before** running `pip install -r requirements.txt`, or reinstall torch
afterwards.

Verify the install:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### Windows troubleshooting: "NumPy requires GCC >= 8.4" / pip tries to build numpy from source

This means pip could not find a **prebuilt wheel** for your Python version/architecture
and fell back to compiling numpy from scratch — which then fails because Windows has
no C compiler configured by default. Common causes and fixes:

1. **Python 3.13 (or newer) with an old numpy/torch lower bound.** numpy only ships
   prebuilt wheels for Python 3.13 starting at **2.1.0**; torch only starting at
   **2.5.0**. `requirements.txt` already reflects this (`numpy>=2.1`, `torch>=2.5`).
   If you still hit this error, check `pip --version` and `python --version` match
   what you expect, and that no cached/pinned older version is being picked up
   (`pip cache purge` if unsure).
2. **32-bit Python.** Check with `python -c "import struct; print(struct.calcsize('P')*8)"`.
   If it prints `32`, uninstall Python and reinstall the **64-bit** installer from
   python.org, then recreate the venv.
3. **Very new Python version with no wheels yet for a smaller dependency** (e.g.
   `thop`, `seaborn`). If one specific package is the failure point rather than
   numpy/torch, it may simply not have wheels for a brand-new Python release yet —
   the safest fix is Python 3.11 or 3.12, which has the widest wheel coverage across
   the whole scientific-Python ecosystem.

To confirm pip is finding wheels (not building from source) before committing to a
full install:
```bash
pip install --only-binary=:all: numpy pandas scikit-learn
```
If that command itself fails, the problem is your Python version/wheel availability,
not the requirements file.

## 3. Get the dataset

Download the **C-MAPSS Turbofan Engine Degradation Simulation** dataset
from the NASA Prognostics Data Repository:
https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/

Unzip it and copy the `train_FD00X.txt`, `test_FD00X.txt`, and
`RUL_FD00X.txt` files (X = 1..4) directly into `data/raw/`.

## 4. Run the pipeline

All commands are run as modules from the **project root** (so relative
imports in `src/` resolve correctly) with the virtual environment active.

```bash
# 1) preprocess one subset (or "all")
python -m src.data.preprocessing --subset FD001

# 2) train the Transformer teacher
python -m src.train_teacher --subset FD001 --device cuda   # or --device cuda

# 3) distil the lightweight student from the trained teacher
python -m src.train_student_kd --subset FD001 --device cuda

# 4) evaluate teacher vs student vs LSTM baseline
python -m src.evaluate --subset FD001 --device cuda

# 5) (optional) compression trade-off sweep across student sizes
python -m src.trade_off_sweep --subset FD001 --device cuda # this one to run for 3
```

Or run steps 1-4 in one go:
```bash
bash scripts/run_pipeline.sh FD001 cpu
```

Repeat with `FD002`, `FD003`, `FD004`, or pass `--subset all` to
`preprocessing.py` / `evaluate.py` to process every subset in one call
(teacher/student training scripts also accept `--subset all` and will
loop through all four).

## 5. Where results land

- `checkpoints/teacher_{subset}.pt`, `checkpoints/student_{subset}.pt`,
  `checkpoints/lstm_baseline_{subset}.pt` — trained weights + recorded
  test RMSE/score.
- `results/comparison_{subset}.csv` — RMSE, NASA score, parameter count,
  model size (MB), FLOPs, and inference time (ms) for teacher, student,
  and baseline, side by side. `results/comparison_all.csv` combines every
  subset.
- `results/tradeoff_sweep_{subset}.csv` — accuracy vs. efficiency for
  each compression configuration defined in `SWEEP_GRID`
  (`src/trade_off_sweep.py`).

## 6. Tuning

Every hyperparameter (teacher/student architecture, distillation loss
weights, window size, RUL cap, batch size, epochs, learning rate) is
centralised in `src/config.py` — edit it there rather than passing extra
CLI flags into the training scripts.

## 7. Notes on the distillation loss

Since RUL prediction is a regression task (not classification),
"soft-target" distillation here means the student is trained to match
the teacher's continuous predicted RUL (temperature-scaled) in addition
to the ground-truth label, and a learnable linear projector aligns the
student's pooled embedding to the teacher's for feature-level
distillation. See `src/distillation/kd_loss.py` for the full formulation
and the in-code citations to the papers this mirrors.
IPEO-Project/
├── README.md                  # goal, setup, how to reproduce each experiment
├── environment.yml            # or pyproject.toml / requirements.txt
├── .gitignore
├── .pre-commit-config.yaml    # ruff, black, nbstripout
├── Makefile                   # make data / train / eval / figures
│
├── configs/                   # one YAML per run = reproducibility
│   ├── base.yaml              # model, optimizer, iterations, crop size, loss
│   ├── spatial/
│   │   ├── all.yaml
│   │   ├── no_africa.yaml
│   │   ├── random_removal.yaml   # matches Africa's size (pixels or fire pixels)
│   │   ├── na_only.yaml
│   │   ├── africa_only.yaml
│   │   ├── region_balanced.yaml
│   │   └── finetune_na.yaml
│   └── temporal/
│       ├── all_years.yaml
│       ├── most_recent_k.yaml
│       ├── random_k.yaml
│       ├── similarity_k.yaml
│       └── single_year.yaml
│
├── data/                      # gitignored (keep only .gitkeep + README)
│   ├── raw/                   # SeasFire down-sampled zarr/nc
│   ├── processed/             # normalized arrays, stats.json
│   └── masks/                 # continent mask, land mask
│
├── src/wildfire/
│   ├── data/
│   │   ├── dataset.py         # crop sampler, train/val/test Dataset
│   │   ├── preprocessing.py   # log transforms, normalization (train stats only)
│   │   ├── masks.py           # land, Africa, North America, Canada
│   │   ├── splits.py          # year/region subsets, random removal
│   │   └── sampling.py        # region-balanced / fire-aware sampling
│   ├── models/
│   │   └── unet.py
│   ├── losses.py              # masked BCE / focal / Dice
│   ├── metrics.py             # AUPRC, F1, IoU, per-region, bootstrap CI
│   ├── train.py               # fixed-iteration training loop
│   ├── evaluate.py            # full-grid inference, per-region metrics
│   ├── baselines/
│   │   ├── climatology.py
│   │   ├── persistence.py
│   │   └── pixel_lgbm.py
│   ├── selection/
│   │   ├── similarity.py      # energy distance / MMD / anomaly stats
│   │   └── year_selection.py  # top-k years, label-free
│   └── utils/                 # config loading, seeding, logging, plotting
│
├── scripts/
│   ├── download_data.py
│   ├── compute_stats.py       # normalization stats from training years
│   ├── run_experiment.py      # --config configs/spatial/no_africa.yaml --seed 0
│   └── aggregate_results.py   # all runs → one CSV of metrics
│
├── notebooks/                 # exploration and analysis only, no core logic
│   ├── 00_data_exploration.ipynb
│   ├── 01_label_alignment_check.ipynb
│   ├── 02_baselines.ipynb
│   ├── 03_threshold_and_loss_tuning_2018.ipynb
│   ├── 04_spatial_analysis.ipynb
│   ├── 05_temporal_analysis.ipynb
│   └── 06_figures_for_report.ipynb
│
├── tests/
│   ├── test_label_alignment.py    # predictors at t vs burned area at t+1
│   ├── test_masks.py
│   ├── test_metrics.py
│   └── test_splits.py             # no leakage, same-size subsets
│
├── results/
│   ├── metrics/               # small CSV/JSON per run (committed)
│   └── figures/
│
├── checkpoints/               # gitignored
└── report/                    # LaTeX or markdown, final figures
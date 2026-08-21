# Prompt: audit per-record / per-fold AAMI class distribution (all classes, not just Q) — local only

Paste this into a fresh Claude Code session in this repo (`ecg-report`). Self-contained — no prior conversation needed. **This is a fully local task — no Azure ML, no cloud job, no data upload.**

## Context

This is a two-stage ECG arrhythmia classifier over MIT-BIH (AAMI classes N/S/V/F/Q). Training uses patient-level `StratifiedGroupKFold` (5 folds) — see `train_binary_stage`/`train_multiclass_stage` in [src/AAMI_classification_multistages/multistage_train.py](../src/AAMI_classification_multistages/multistage_train.py). The evaluation job measured **extreme per-fold variance** in multiclass validation accuracy (0.22–0.98 across the 5 folds, same model/pipeline) — see `docs/next_steps_prompt.md` and the "Model Evaluation" section of `README.md` for the full picture.

**Root cause found for class Q specifically**, using the script described below: 4 records — **102 (2084), 107 (2077), 104 (2063), 217 (1802)** — hold 8026 of 8041 total Q windows (99.8%) across all folds/splits combined. The rest of the loaded patients hold at most 5 each, most hold zero. Because `StratifiedGroupKFold` splits by patient, whichever fold each of those 4 "carrier" patients lands in swings that fold's Q representation wildly. This alone plausibly explains a large share of the fold-to-fold accuracy swing, since Q is the second-largest class overall and accuracy is support-weighted.

**What's not yet known:** whether classes S, V, and F show the same concentration pattern, or whether Q is uniquely extreme. This prompt generalizes the one-off Q analysis into a reusable, visual audit across **all four anomaly classes**.

## Everything needed already exists locally — reuse it, don't reinvent

- **[notebooks/q_distrib.py](../notebooks/q_distrib.py)** already does exactly this analysis, but hardcoded to class `'Q'` only. It: loads `fold_splits_multiclass.json`, for every `(fold, split, record_id)` triple counts windows via `extract_AAMI_windows(...)` from [multistage_preprocessing.py](../src/AAMI_classification_multistages/multistage_preprocessing.py) (the *same* window-extraction function the training pipeline itself uses — keep using this, not a raw-annotation count, so the numbers match what the model actually trains/validates on), and writes `q_per_fold.csv`, `q_per_patient.csv`, `q_top_contributor_per_fold_split.csv` plus a bar chart to `notebooks/outputs/`. **Generalize this script to loop over all AAMI classes (S, V, F, Q — the anomaly classes used by the multiclass stage) instead of hardcoding Q.** Keep its CLI shape (`--data-dir`, `--fold-splits`, `--output-dir`) and its per-fold/per-split/per-record/share_pct structure — just parametrize the class.
- **[notebooks/fold_splits_multiclass.json](../notebooks/fold_splits_multiclass.json)** is already sitting locally — no need to fetch it from anywhere. (There is no local `fold_splits_binary.json` yet; scope this task to the multiclass stage only unless you copy that file in too.)
- **Local MIT-BIH data**: `notebooks/loader.ipynb` shows the path already used on this machine: `C:\Users\huber\Downloads\mit-bih-arrhythmia-database-1.0.0\mit-bih-arrhythmia-database-1.0.0`. Verify it still exists before assuming it as the default `--data-dir`.
- **[src/evaluation/generate_report.py](../src/evaluation/generate_report.py)**: `plot_confusion_matrix_figure` and `plot_per_fold_metrics_figure` show this project's matplotlib style (plain, labeled, `plt.tight_layout()`) — match it for visual consistency, but do **not** import from or depend on `generate_report.py` itself (it pulls in `mlflow`/`azure-ai-ml`, which this task has no reason to need).

## Task

1. **Extend `notebooks/q_distrib.py`** (or split it into a new `notebooks/class_distribution_audit.py` if that reads cleaner — your call) to compute, for every AAMI anomaly class (S, V, F, Q):
   - the same per-fold/per-split/per-record window counts `q_distrib.py` already produces for Q alone
   - a **concentration metric**: what % of that class's total windows are held by its top-1, top-3, top-5 "carrier" records (this is what quantified the Q finding — 4 records = 99.8% — and needs to exist for S, V, F too)
   - flag any `(fold, split, class)` combination with count `< 30` (same floor already used as the IQR-downsampling minimum in `load_data`, `multistage_train.py`) as statistically degenerate
2. **Generate figures**, saved to `notebooks/outputs/`:
   - a **heatmap**: records (rows) × AAMI classes (columns), cell = window count, log-scale color (Q's spread is ~2000 vs. single digits for others)
   - a **per-fold grouped bar chart** of validation window counts per class across the 5 folds
   - a **concentration chart** per class: top-N records' % share of that class's total, N=1..5
3. **Write a markdown report**, `notebooks/outputs/class_distribution_report.md`, embedding the figures (relative paths) and the CSVs' key numbers, with a written conclusion: *is the fold-variance problem specific to Q, or does it affect S/V/F too, and which folds are trustworthy per class as a result?* Cross-reference `README.md`'s "Model Evaluation" section (fold 2 is the worst multiclass performer — check whether fold 2 is also where several classes' carrier patients clustered together, not just Q's).
4. **Read-only diagnostic** — do not change the training pipeline, the CV splitting logic, or anything under `src/`.

## Verification

Sanity-check your generalized script's Q output against the numbers already confirmed: **102→2084, 107→2077, 104→2063, 217→1802, total Q=8041** (summed across folds/splits — note `q_distrib.py`'s `q_per_fold_df` counts each record once per fold it appears in as train or val, so watch for double counting when computing the *overall* per-record total; `q_per_patient_df`, already deduplicated by `record_id`, is the one to check against these numbers). If it doesn't match, something in the counting logic broke — fix that before trusting the S/V/F numbers.

## Practical notes

- Fully local. No Azure ML, no `mlflow`, no data upload — everything needed (data, fold splits, code) is already on this machine.
- Reuse the existing `requirements.txt` environment (`pandas`, `numpy`, `wfdb`, `matplotlib` are already there); no new dependencies should be needed.

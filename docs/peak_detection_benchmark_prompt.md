# Prompt: benchmark neurokit2 R-peak detectors against the custom Pan-Tompkins implementation — local only

Paste this into a fresh Claude Code session in this repo (`ecg-report`, branch `peak_detection`). Self-contained — no prior conversation needed. **Fully local — no Azure, no cloud job.**

## Context

This project has a custom Pan-Tompkins R-peak detector at [src/pan_tompkins.py](../src/pan_tompkins.py) (`pan_tompkins_detect(signal, fs)`), used by both the FastAPI inference service and the evaluation pipeline (`src/cascade.py`). It was validated against 10 MIT-BIH records with tolerance ±5 samples (see `README.md`'s "Pan-Tompkins Detector Validation" section): aggregate sensitivity 0.884, PPV 0.956 — but two records, **104** (sensitivity 0.546) and **108** (sensitivity 0.503), are much worse than the rest (~0.97+). A more recent full-cascade evaluation (37 patients, see `README.md`'s "Cascade End-to-End Results") found the detector now accounts for **36.6% of all lost anomalies** in the cascade — the single largest loss category, larger than either classification stage's own errors. Improving R-peak detection is currently the highest-leverage lever identified for this project.

**The question this prompt answers**: does any of the 17 R-peak detection algorithms bundled in the `neurokit2` library beat the custom Pan-Tompkins implementation - in aggregate, and specifically on the two known-weak records (104, 108) - closely enough to be worth adopting?

## Environment - read this before running anything

`neurokit2` (v0.2.13) is installed in the conda environment **`dl_env`** at `D:\conda\envs\dl_env`, **not** in whatever interpreter this project's own `requirements.txt`/`conda.yml` normally uses. Any script that imports `neurokit2` must run with:

```
D:\conda\envs\dl_env\python.exe your_script.py
```

(or the dl_env equivalent if activated). Verify with `D:\conda\envs\dl_env\python.exe -c "import neurokit2; print(neurokit2.__version__)"` before writing code against it. This project's own `src/pan_tompkins.py` and `src/cascade.py` only need `numpy`/`scipy`/`wfdb`, already available in the environment this repo normally uses - so the comparison script needs to run under `dl_env` specifically because `neurokit2` is the new dependency, not because anything else changed.

**Local MIT-BIH data**: `C:\Users\huber\Downloads\mit-bih-arrhythmia-database-1.0.0\mit-bih-arrhythmia-database-1.0.0` (already used by `notebooks/loader.ipynb` and `notebooks/q_distrib.py` - verify it still exists before assuming it as the default `--data-dir`).

## Relevant existing code to reuse, not reinvent

- **`pan_tompkins_detect(signal, fs, return_intermediate=False)`** in [src/pan_tompkins.py](../src/pan_tompkins.py) - the baseline to beat. Returns a 1D array of detected R-peak sample indices.
- **`cascade.match_detected_to_annotated_peaks(detected_peaks, annotated_samples, tolerance_samples=5)`** in [src/cascade.py](../src/cascade.py) - already implements the exact matching logic (nearest-neighbor within tolerance, no double-matching) used for the README's Pan-Tompkins validation numbers. Reuse this for every method being compared, so results are apples-to-apples with the numbers already in `README.md`.
- **`evaluate_pan_tompkins`** in [src/evaluation/generate_report.py](../src/evaluation/generate_report.py) (around line 305) - shows the established pattern: read `.atr` annotations via `wfdb.rdann`, filter to symbols with a known AAMI class via `SYMBOL_TO_CLASS` (from `src/AAMI_classification_multistages/multistage_preprocessing.py`), read the signal via `wfdb.rdsamp`, detect, match, compute per-record and aggregate sensitivity/PPV. Follow this exact methodology (same tolerance, same annotation filtering) for the neurokit2 methods too, or the comparison isn't valid.
- **`get_record_ids(data_dir)`** in `multistage_preprocessing.py` - lists available records in a data directory.

## neurokit2 API notes (already checked against the installed v0.2.13, don't re-derive)

```python
import neurokit2 as nk
signals, info = nk.ecg_peaks(raw_signal, sampling_rate=fs, method="pantompkins1985")
r_peaks = info["ECG_R_Peaks"]  # 1D array of sample indices, same shape/meaning as pan_tompkins_detect's return
```

Valid `method` strings (17 algorithms + 1 ensemble; use the primary name, not the deprecated alias, though aliases work too):
`neurokit` (default), `pantompkins1985`, `hamilton2002`, `christov2004`, `engzeemod2012`, `elgendi2010`, `kalidas2017`, `martinez2004`, `rodrigues2021`, `gamboa2008`, `nabian2018`, `zong2003`, `manikandan2012`, `khamis2016`, `emrich2023`, `koka2022`, and `promac` (probabilistic ensemble of all of the above - expect it to be markedly slower per record).

`ecg_peaks` accepts the raw signal directly (no separate `ecg_clean` call required - each method applies its own internal preprocessing) - confirm this empirically on one record before assuming it for all 18 methods, since a couple of the less common ones may behave differently or throw on unexpected input; wrap each method's call in a try/except so one failing method doesn't kill the whole sweep.

## Task

1. **Write a local comparison script** (e.g. `notebooks/peak_detection_benchmark.py`, following the existing `notebooks/q_distrib.py` / `notebooks/class_distribution_audit.py` convention: plain `.py` script, `argparse` with `--data-dir` defaulting to the path above, results under `notebooks/outputs/`). It must run under `D:\conda\envs\dl_env\python.exe`.
2. For **every MIT-BIH record** in `--data-dir` (all of them - this is local and free, no reason to sample only 10 like the Azure job does to control cost) and **every method** (18 total: `pan_tompkins_detect` as the baseline + the 17 neurokit2 algorithms, `promac` optional/flagged separately given its cost):
   - detect R-peaks
   - match against annotated peaks via `cascade.match_detected_to_annotated_peaks` (tolerance ±5 samples, matching the existing methodology exactly)
   - compute sensitivity and PPV per record
   - also record wall-clock detection time per record (methods vary hugely in cost - `promac` especially - and this project already tracks CPU benchmarks elsewhere, so capture it here too)
3. **Aggregate per method**: overall sensitivity/PPV (pooled across all records, not averaged per-record - matches `evaluate_pan_tompkins`'s existing aggregation), plus **sensitivity specifically on records 104 and 108** (the two known-weak cases - this is the number that actually matters for the motivating question).
4. **Write a report** `notebooks/outputs/peak_detection_benchmark_report.md`: a table of all 18 methods × (aggregate sensitivity, aggregate PPV, sensitivity on 104, sensitivity on 108, mean detection time per record), sorted by aggregate sensitivity, plus a bar chart (`notebooks/outputs/peak_detection_comparison.png`) comparing methods on aggregate sensitivity and on the 104/108 subset side by side.
5. **Recommend the best method**, with an explicit rationale that weighs: (a) does it clearly beat `pan_tompkins_detect`'s aggregate 0.884/0.956, (b) does it specifically fix the 104/108 weakness (the actual motivation for this whole benchmark - a method that's marginally better on average but no better on 104/108 hasn't solved the problem that mattered), (c) is its per-record detection time reasonable for the FastAPI service's latency budget (see `README.md`'s CPU Inference Benchmarks - Pan-Tompkins currently takes ~0.04s/record; note if a candidate is far slower, since that's a real deployment cost, not just an evaluation-time one).

## What this prompt does NOT ask for

Don't touch `src/pan_tompkins.py`, `src/cascade.py`, or the FastAPI service in this pass - this is a read-only local benchmark to decide *whether* switching detectors is worth it. If a clear winner emerges, swapping it into the production code path is a deliberate follow-up decision, not an automatic next step here.

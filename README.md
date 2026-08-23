# ECG Report AI

## Overview

ECG Report AI is an end-to-end Machine Learning project focused on automated ECG (Electrocardiogram) classification using Deep Learning techniques.

The project was built to demonstrate practical Machine Learning Engineering skills across the complete ML lifecycle, including model development, experimentation, validation, cloud-assisted training, containerized deployment, CI/CD automation, and model lifecycle management.

Rather than focusing solely on model performance, the project emphasizes reproducibility, maintainability, deployment readiness, and industry-standard MLOps practices — including honestly reporting what the system does and does not yet do well (see [Known Limitations](#known-limitations)).

The system analyzes ECG recordings from the MIT-BIH Arrhythmia Database and serves predictions through a Streamlit application connected to a containerized inference API.

> **Evaluation snapshot: 2026-08-21.** All numbers below come from a real evaluation job run against the live cascade (R-peak detection → binary → multiclass) on held-out MIT-BIH patients — not training-time metrics, and reported with 95% bootstrap confidence intervals (resampled by patient, not by beat) rather than bare point estimates. The R-peak detector was switched from a custom Pan-Tompkins implementation to `neurokit2` in this snapshot (see [R-Peak Detector Choice](#r-peak-detector-choice)); this is the single biggest driver of the numbers below vs. earlier snapshots.
>
> **This snapshot predates the exclusion of AAMI class Q.** That change is decided and documented in [AAMI class scope](#aami-class-scope-why-q-paced-beats-are-excluded) but not yet reflected in any measurement here — the numbers are kept so the before/after is visible rather than silently overwritten.

---

# Project Objectives

The main goals of the project are:

* Build an automated ECG classification system.
* Develop robust binary and multiclass classification models.
* Apply cross-validation for reliable evaluation.
* Implement Champion vs Challenger model selection.
* Create a reproducible ML workflow.
* Deploy inference through containerized services.
* Utilize Azure for scalable model training and comparison.
* Explore synthetic ECG generation to improve generalization.

---

# Key Features

## Machine Learning

* Binary ECG classification (Normal vs. anomaly)
* Multiclass ECG classification (AAMI classes S/V/F — see [AAMI class scope](#aami-class-scope-why-q-paced-beats-are-excluded))
* Deep Learning models built with TensorFlow/Keras (Inception-Conformer architecture)
* Custom Pan-Tompkins R-peak detector, shared between the inference service and the evaluation pipeline
* Cross-validation-based evaluation (patient-level `StratifiedGroupKFold`)
* Model comparison framework (Champion vs Challenger)
* Hyperparameter experimentation
* Synthetic data augmentation using WGAN-GP (see [WGAN-GP Case Study](#wgan-gp-case-study))

## MLOps

* CI/CD pipeline
* Automated validation
* Dockerized inference
* Experiment tracking (MLflow, via Azure ML)
* Model versioning
* Champion vs Challenger workflow, computed inside the Azure ML evaluation job

## Cloud Integration

* Azure-based training and evaluation workflows
* Model artifact storage
* Centralized model comparison

## Application Layer

* Streamlit frontend
* FastAPI inference backend
* Containerized prediction service

---

# Dataset

The project uses the MIT-BIH Arrhythmia Database, one of the most widely recognized benchmark datasets for ECG classification research.

Dataset characteristics:

* Expert-annotated ECG recordings
* Clinically validated labels
* Public research benchmark
* High-quality and clean ECG signals

Because the dataset already contains clean ECG recordings, no advanced signal denoising or signal processing pipeline was required. The project focuses primarily on machine learning, model evaluation, deployment, and MLOps engineering.

## AAMI class scope: why Q (paced) beats are excluded

The AAMI standard defines five beat classes — N, S, V, F and Q — and this project originally targeted all five. **Class Q is now excluded from both training stages.** All patients are kept; only their Q-annotated beats are dropped, so the classifier still accepts recordings from any patient and simply does not attempt to assign those beats an arrhythmia label.

What class Q actually contains in MIT-BIH, counted across the whole database:

| Annotation symbol | Meaning | Count | Share of class Q |
| --- | --- | ---: | ---: |
| `/` | Paced beat | 9,056 | 89.4% |
| `f` | Fusion of paced and normal beat | 1,038 | 10.2% |
| `Q` | Genuinely unclassifiable | 33 | 0.3% |

So "class Q" is, in practice, **pacemaker rhythm** — not an arrhythmia. It says something about the device implanted in the patient, not about the pathology the system is meant to detect. Three measured findings drove the decision:

* **It was never a 5-fold problem.** 99.8% of Q beats come from just four recordings (102, 104, 107, 217 — the paced records). Every CV fold's Q validation set turned out to be ~100% one patient, making Q evaluation a 4-patient leave-one-out disguised as 5-fold CV, with one fold left holding just 2 Q windows.
* **It failed exactly as that structure predicts.** Folds validating on patients 102/104/107 reached Q F1 of 1.00 / 0.85 / 0.64; the fold validating on patient 217 collapsed to 0.03, assigning 1,692 of its 1,811 true Q beats to class V. The model was learning one pacemaker's morphology, not a general concept.
* **It dominated the fold-to-fold variance** that made every other experiment in this project unmeasurable — GAN augmentation and class weighting both produced effects far smaller than the noise Q introduced.

Mapping Q to Normal instead of dropping it was considered and rejected: paced beats have a wide QRS morphologically close to class V, so labelling them Normal would push the same confusion down into Stage 1, where errors are unrecoverable. Excluding the four paced records entirely (the AAMI EC57 recommendation) was also considered, but dropping only the Q beats keeps every patient in the dataset, which better matches how the deployed system is meant to behave.

> Note that beats excluded from the label set are still detected by Pan-Tompkins at inference time — the system does not refuse them, it simply has no arrhythmia class for them. The cascade evaluation excludes them from ground truth, consistent with AAMI practice of excluding paced beats from performance assessment.

**The evaluation numbers below were measured before this change, with Q included, and will be replaced once the first post-change training and evaluation runs complete.** They are kept for now so the effect of the change is visible rather than silently overwritten.

---

# System Architecture

```text
                MIT-BIH Dataset
                        │
                        ▼
              Pan-Tompkins R-peak Detection
                        │
                        ▼
             Stage 1: Binary Classifier
              (Normal vs. Anomaly)
                        │
              anomaly?  ▼  (only anomalies proceed)
             Stage 2: Multiclass Classifier
                (AAMI: S / V / F — Q excluded,
                 see "AAMI class scope" above)
                        │
                        ▼
                Cross-Validated Training
                (patient-level StratifiedGroupKFold)
                        │
                        ▼
          Azure ML Evaluation Job (cascade metrics,
           baselines, benchmarks, promotion decision)
                        │
                        ▼
              Champion vs Challenger
                        │
                        ▼
                Model Artifact Store
                        │
                        ▼
                 Docker Container
                        │
                  FastAPI Inference
                        │
                        ▼
                Streamlit Frontend
```

---

# Technology Stack

## Machine Learning

* Python
* TensorFlow / Keras
* Scikit-Learn
* NumPy / Pandas
* SciPy (Pan-Tompkins signal processing)

## MLOps & DevOps

* Git
* GitHub
* GitHub Actions (CI only — lint/tests, no Azure calls)
* Docker
* CI/CD Pipelines

## Cloud

* Microsoft Azure Machine Learning
* Azure Storage

## Application

* Streamlit
* FastAPI

## AI-Assisted Development

* Claude Code and Gemini Code Assist

Claude Code and Gemini Code Assist were used as productivity tools during development for code generation, refactoring support, documentation assistance, and development acceleration. All architectural decisions, model design choices, experimentation, and implementation remained under developer supervision.

Beyond code generation, Claude Code was used hands-on to design and build the Azure ML evaluation pipeline described in this README (`src/evaluation/generate_report.py`, `src/cascade.py`, the fold-split persistence in `multistage_train.py`) and to diagnose and fix a series of real issues hit while getting it running end-to-end on Azure, including: a subscription vCPU quota mismatch on `cpu-cluster` (`max_instances` sized beyond the Standard DSv2 quota), an invalid `azureml://jobs/...` URI format for mounting a prior job's outputs (fixed by downloading job outputs locally instead), an import bug (`load_data` pulled from the wrong module), a missing `fold_splits_binary.json` caused by the training job only running `--task multiclass` instead of `--task both`, and a broken `mlflow`/`azureml-mlflow` artifact-store plugin (`azure-ai-ml` was also missing from `conda.yml` entirely, which silently broke the in-job Champion lookup). All of these were found by actually running the pipeline against live Azure ML jobs, not by inspection alone.

---

# Training Workflow

The model development process follows a structured workflow:

1. Dataset preparation
2. Cross-validation training (binary + multiclass, both stages in one job)
3. WGAN-GP minority-class augmentation (multiclass stage only)
4. Cascade evaluation on held-out patients (Azure ML evaluation job)
5. Champion vs Challenger comparison (computed inside the evaluation job)
6. Model artifact registration (local, interactive — see [MLOps Pipeline](#mlops-pipeline))
7. Deployment to inference environment

This approach ensures consistent evaluation and repeatable experimentation.

---

# Model Evaluation

## Cascade End-to-End Results (headline metric)

This is the number that matters most: running the *actual* production pipeline (R-peak detection → Stage 1 binary → Stage 2 multiclass) on held-out MIT-BIH patients, matching detected peaks back to ground-truth annotations (±5 samples).

Earlier snapshots of this section evaluated only 9 (then 5) held-out records, because the binary and multiclass cross-validation folds are computed independently and a patient held out of *both* stages was found only by intersecting fold indices that happened to match — mostly by luck. That was fixed by pairing each patient with the specific binary/multiclass model pair that actually never saw them, regardless of fold index (`src/evaluation/fold_pairing.py`), recovering the full eligible population (37 patients, used ever since).

| Metric | Value |
| --- | --- |
| **Cascade macro F1** | **0.3444** (95% CI: 0.2391–0.4385) |
| Cascade weighted F1 | 0.7951 |
| Cascade accuracy | 0.8029 |
| Records evaluated | 37 |

Per-class breakdown at the cascade output:

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| N | 0.8845 | 0.9113 | 0.8977 | 63,956 |
| S | 0.0309 | 0.0028 | 0.0052 | 1,056 |
| V | 0.2997 | 0.4322 | 0.3540 | 5,564 |
| F | 0.0000 | 0.0000 | 0.0000 | 778 |
| Q | 0.5978 | 0.3807 | 0.4652 | 8,043 |

All five classes clear the 30-example support floor used elsewhere in this project (`--min-support-floor` in `generate_report.py`), so macro F1 above isn't being distorted by a class measured on a handful of examples — a real risk at the smaller record counts of earlier snapshots.

**Where true anomalous beats are lost** (15,441 true anomalies total across the 37 patients):

| Outcome | Count | Share |
| --- | --- | --- |
| Missed by the R-peak detector | 3,707 | 24.0% |
| Detected, but classified Normal by Stage 1 | 3,907 | 25.3% |
| Reached Stage 2, but misclassified | 2,357 | 15.3% |
| Correctly classified end-to-end | 5,470 | **35.4%** |

Over a third of true anomalous beats now survive the full cascade correctly, up from ~1 in 5 in every earlier snapshot — the biggest single jump measured in this project, driven almost entirely by switching the R-peak detector (see [R-Peak Detector Choice](#r-peak-detector-choice)). With detection loss cut roughly in third, **Stage 1 false negatives are now the single largest remaining loss category** (25.3%, edging out the detector's remaining 24.0%) — the bottleneck priority named at the very start of this project's model-quality work is, again, the next lever to pull.

![Confusion matrix](docs/img/confusion_matrix.png)

---

## Stage 1 — Binary (Normal vs. Anomaly)

Clinical metrics, pooled across all 5 CV folds:

| Metric | Value |
| --- | --- |
| Sensitivity (recall on anomalies) | **0.6578** |
| Specificity | 0.9024 |
| PPV | 0.4982 |
| NPV | 0.9471 |

These numbers are measured on held-out annotated windows directly, independent of which R-peak detector the cascade uses at inference time - unlike the cascade section above, switching detectors doesn't move them (confirmed: identical to the last pre-switch evaluation on the same training run, to 4 decimal places). Stage 1 sensitivity is the ceiling on the *classification* side of the cascade: every anomaly missed here can never reach Stage 2. It was 0.5768 before fixing how the binary-stage cross-validation split is stratified. `StratifiedGroupKFold` was splitting on the flattened Normal-vs-anomaly label, which is blind to AAMI subtype; that let one CV fold accidentally concentrate ~60% of all class S beats (supraventricular ectopic — morphologically the hardest to distinguish from Normal) in its validation set, forcing a near-zero decision threshold just to detect them. Re-stratifying by AAMI subtype instead fixed this: sensitivity, specificity, and PPV all improved *simultaneously* (not a sensitivity/specificity trade-off), and no CV fold's AUC is catastrophically low anymore.

**Recalibrating the decision threshold away from 0.5 was tried and rejected — twice, on two independent training runs.** Two approaches were tested each time: (1) picking a threshold that hits 90% Stage 1 sensitivity in isolation, and (2) sweeping candidate thresholds through the *full* cascade and ranking by the resulting cascade macro F1. Both were measured against the real cascade output, not just Stage 1 metrics — approach (1) reliably *hurts* cascade macro F1 (0.3444 → 0.2744 in the current snapshot) by flooding Stage 2 with false positives from the resulting collapse in specificity; approach (2)'s best candidate (threshold 0.7 this time, 0.3 or 0.5 in earlier snapshots — the "winner" itself isn't stable run to run) beat the 0.5 default by less than its own bootstrap confidence interval both times, i.e. within noise. The default threshold of 0.5 stays.

## Stage 2 — Multiclass (AAMI: S / V / F / Q — pre-exclusion snapshot)

Aggregate classification report across all 5 folds (window-level, not the cascade — see above for the end-to-end number):

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| S | 0.4283 | 0.4151 | 0.4216 | 648 |
| V | 0.5204 | 0.7966 | 0.6295 | 4,765 |
| F | 0.0000 | 0.0000 | 0.0000 | 103 |
| Q | 0.8121 | 0.5682 | 0.6686 | 8,041 |

| Metric | Value |
| --- | --- |
| Macro F1 | 0.4299 |
| Weighted F1 | 0.6380 |
| Accuracy | 0.6369 |
| Mean CV AUC | 0.8680 |

Like Stage 1's clinical metrics above, this section is window-level and detector-independent (identical before/after the R-peak detector switch).

**Per-fold breakdown — variance is still the dominant story here.** Only the binary stage's CV stratification was fixed (see Stage 1 above); the multiclass split is unchanged and shows it:

| Fold | Val Acc | Val AUC | Macro F1 | Weighted F1 |
| --- | --- | --- | --- | --- |
| 0 | 0.7829 | 0.9107 | 0.4763 | 0.7600 |
| 1 | 0.9756 | 0.9977 | 0.6483 | 0.9720 |
| 2 | 0.2945 | 0.7847 | 0.1896 | 0.2548 |
| 3 | 0.6044 | 0.6700 | 0.2512 | 0.7093 |
| 4 | 0.9847 | 0.9766 | 0.5177 | 0.9860 |

Val accuracy ranges from 0.29 to 0.98 across 5 folds of the *same* model and pipeline. Against majority-class and stratified-random baselines, the model clearly beats both in 4 of 5 folds — but **loses to both baselines in fold 2** (model macro F1 0.190 vs. baseline 0.218/0.226), which the aggregate numbers above would otherwise hide. A local audit (`notebooks/class_distribution_audit.py`) traced this to the same root cause as the binary fix above, not yet applied to multiclass: a handful of "carrier" patients hold almost all of a given class's examples (e.g. 4 patients hold 99.8% of all class Q windows), so which fold they land in swings that fold's class balance heavily. Fold 2 specifically got carrier patients for *two* classes (Q and S) at once.

![Per-fold metrics](docs/img/per_fold_metrics.png)

---

## Pan-Tompkins Detector Validation

Extended from an earlier 2-record spot check to 10 MIT-BIH records, tolerance ±5 samples:

| Metric | Value |
| --- | --- |
| Aggregate sensitivity | 0.8840 |
| Aggregate PPV | 0.9556 |

Per-record, most are ≥0.97 sensitivity, but two records are notably worse — record 104 (0.5464) and record 108 (0.5026) — both known in the MIT-BIH literature for atypical/noisy morphology, suggesting the detector's fixed adaptive-threshold parameters don't generalize to every recording condition.

These same two records turned out to matter more than this 10-record spot check alone suggested: once the [cascade evaluation](#cascade-end-to-end-results-headline-metric) covered its full eligible patient population instead of a small lucky subset, missed detections became (at the time) the single largest category of lost anomalies (36.6%) — larger than either classification stage's own errors. That finding directly motivated the detector comparison below.

![Pan-Tompkins detection](docs/img/pan_tompkins_detection.png)

---

## R-Peak Detector Choice

The custom Pan-Tompkins implementation above was replaced as the cascade's default R-peak detector after a three-stage comparison, all local (no Azure cost) except the final confirmation:

1. **Isolated benchmark** — 48 MIT-BIH records × 18 candidates (the custom implementation + all 17 algorithms bundled in `neurokit2`, `docs/peak_detection_benchmark_prompt.md`, `notebooks/outputs/peak_detection_benchmark_report.md`). `neurokit2`'s own `'neurokit'` method won on aggregate sensitivity (0.92 vs. 0.87) and nearly fixed record 104 (0.55 → 0.95) at comparable latency (~70ms vs. ~46ms/record) — but no method, including a 10-algorithm ensemble (`'promac'`), fixed record 108 (best achieved there: 0.67, still weak).
2. **Cascade-level, patient-paired bootstrap** — isolated detector accuracy doesn't guarantee the cascade improves (this project has measured the opposite before, see Stage 1's threshold-calibration story). On a first training run, `'neurokit'`'s cascade-level advantage was suggestive but inconclusive (+0.015 macro F1, 95% CI [-0.003, 0.038], P=0.94). Replicated on a second, independent training run, the same comparison reached full statistical significance: **+0.035 cascade macro F1, 95% CI [0.003, 0.084], P=1.00**. `'promac'` was never reliably better than plain `'neurokit'` despite a similar point estimate, at 150–300× the latency, and still didn't fix record 108 (0.45, actually the worst of the three) — rejected.
3. `neurokit_detect` ([src/pan_tompkins.py](src/pan_tompkins.py)) is now `src/cascade.py`'s default `peak_detector` (dependency-injected, `pan_tompkins_detect` still available by passing it explicitly), used by both the FastAPI service and the evaluation pipeline since they share this code path. The exact same swap in the offline evaluation script reproduced the local experiment's cascade macro F1 to 4 decimal places, confirming the production wiring is correct.

This is the single biggest driver of the current [Cascade End-to-End Results](#cascade-end-to-end-results-headline-metric) above.

---

## CPU Inference Benchmarks

Measured on the training/evaluation compute (`Standard_DS3_v2`, 4 vCPU, CPU-only):

| Metric | Value |
| --- | --- |
| Mean Pan-Tompkins detection time | 0.041 s/record |
| Mean Stage 1 (binary) inference time | 1.314 s/record |
| Mean Stage 2 (multiclass) inference time | 0.485 s/record |
| Per-beat latency (mean) | 0.92 ms |
| Per-beat latency (p95) | 1.29 ms |
| **Batched vs. per-window `predict()` speedup** | **~88x** |
| Binary model parameters | 438,081 |
| Multiclass model parameters | 438,468 |
| Binary model size | 5.4 MB |
| Multiclass model size | 5.4 MB |

The ~88x batching speedup quantifies the payoff of refactoring `run_stream` from sequential to batched inference in the FastAPI service. Detector timing here still profiles the legacy Pan-Tompkins implementation specifically (the benchmarking code wasn't updated when the cascade's default detector changed); `neurokit_detect`'s measured latency is ~70ms/record from the isolated benchmark in [R-Peak Detector Choice](#r-peak-detector-choice) — still a small fraction of Stage 1/2's own inference time.

---

## WGAN-GP Case Study

WGAN-GP (Conv1DTranspose generator/critic, gradient penalty) tops up rare AAMI classes during multiclass training. The first verdict on it was "inconclusive"; that turned out to be an artefact of the experiment, not a property of the technique, and re-running it properly produced a clear negative result.

### The generator itself is good

Before asking whether augmentation helps downstream, the generator was evaluated directly — a fast local harness trains one WGAN-GP per class/fold and scores the synthetic beats against the real ones (`Azure/local-run/run_gan_experiment_local.py`):

| Metric | Class F (10k epochs) | Class S (10k epochs) | Reading |
| --- | ---: | ---: | --- |
| Mean-beat Pearson r | 0.9982 | 0.9980 | near-identical average morphology |
| Mean-beat MAE | 0.0265 | 0.0159 | ~0.5% of signal range |
| Mann-Whitney p (per-beat RMS) | 0.764 | 0.150 | no detectable RMS difference |
| RBF MMD | 0.0080 | 0.0049 | distributions closely matched |

FFT magnitude matches across the whole band (no high-frequency artefacts), PCA shows synthetic beats covering the real distribution with no clustering, and real-to-synthetic DTW distance ≈ real-to-real distance, which rules out memorisation of the training beats.

### But the augmentation does not help the classifier

The original "+5% growth per class" cap was, on inspection, arithmetically a no-op: the cap scales with the class's *own* size, so class F received **3–4 synthetic windows per fold**. Raised to +100% (≈600 synthetic windows per fold, a ~20× increase) and compared against a `--skip-gan-augmentation` run on **identical fold splits**:

| Metric | With GAN | Without GAN |
| --- | ---: | ---: |
| Mean Val Accuracy | 0.6831 | **0.7721** |
| Mean Val AUC | 0.8587 | **0.8776** |
| Mean F1 | 0.4086 | **0.4566** |

Worse on every aggregate metric and in 4 of 5 folds individually. Class F stayed at 0.00 F1 in 4 of 5 folds despite having twice the data.

**Verdict: at this dataset size, WGAN-GP augmentation does not improve the downstream classifier**, even with a generator that passes every distributional and morphological check thrown at it and an augmentation dose 20× larger than originally used. The bottleneck for rare classes is not the quantity of examples.

![Real vs synthetic ECG beats](docs/img/wgan_real_vs_synthetic.png)

---

# Champion vs Challenger Strategy

A Champion vs Challenger framework supports continuous model improvement.

## Champion

The currently registered best-performing model (`ecg_multiclass_model` in the Azure ML model registry).

## Challenger

A newly trained candidate model, evaluated against the Champion by the Azure ML **evaluation job itself** — not by a local or CI script.

The evaluation job computes `cascade_macro_f1` (the end-to-end metric above, not just per-window CV AUC) for the challenger, looks up the current Champion's score, and writes `promotion_decision.json` with the verdict and reasoning. This runs inside the job, authenticated via the compute cluster's own managed identity — never with a local or CI credential.

This strategy reflects real-world Machine Learning deployment practices where model replacement requires objective, end-to-end validation rather than relying solely on training-time metrics.

---

# MLOps Pipeline

The full loop is **Continuous Training (CT) → Evaluation → Continuous Delivery (CD)**, and it is split across two trust boundaries because of a hard constraint: **the Azure subscription backing this project is a student subscription that does not support service-principal credentials**, so nothing outside Azure can authenticate to the workspace headlessly.

* **CT and Evaluation run entirely on Azure ML**, submitted by local orchestration scripts that are intentionally not tracked in this repo (they're workspace-specific — see `.gitignore`). The evaluation job also computes the Champion vs Challenger decision in-job, using the compute's managed identity — no interactive login needed there.
* **CD is local and interactive** (`src/evaluate_and_register.py`): it reads `promotion_decision.json` from the evaluation job's outputs using `InteractiveBrowserCredential` (a real browser login, run by a person), and if promoted, registers the new Champion model.
* **GitHub Actions stays limited to code CI** — lint and tests only. It never authenticates to Azure, by design (the commented-out CT step in `.github/workflows/mlops.yml` reflects an earlier, abandoned attempt at a service-principal-based GitHub Actions flow).
* The PR step (auto-branch + `gh pr create`) also runs locally, immediately after a successful CD, using the developer's own `git`/`gh` credentials.

This is not a workaround to be fixed later — it's the correct shape of the pipeline given the auth constraint, and it mirrors how a team without org-wide service-principal access would actually have to run this in production.

---

# Model Versioning

The project includes model versioning based on Keras model artifacts, registered in the Azure ML model registry and tagged with the metrics that justified promotion (`cascade_macro_f1`, `cv_mean_val_auc`, `binary_sensitivity`, `binary_specificity`).

Benefits:

* Traceability
* Reproducibility
* Rollback capability
* Easier deployment management

---

# Azure Integration

Microsoft Azure ML is used as a training, evaluation, and experimentation platform.

Azure responsibilities:

* Running training and evaluation workloads on `cpu-cluster` (`Standard_DS3_v2`, CPU-only, capped at `max_instances=1` to respect the subscription's 6-vCPU DSv2 family quota)
* Comparing model candidates (Champion vs Challenger, computed in-job)
* Storing trained model artifacts and evaluation reports
* MLflow experiment tracking, native to Azure ML

Inference is intentionally separated from Azure and handled through Docker-based services, enabling deployment flexibility and infrastructure independence.

---

# Containerized Inference

The prediction engine operates inside a Docker container.

Inference workflow:

1. User uploads ECG data through Streamlit.
2. Streamlit sends a request to the API.
3. The FastAPI service runs the same cascade logic (`src/cascade.py`) used by the evaluation job — Pan-Tompkins detection, batched Stage 1, batched Stage 2 on anomalies only.
4. Results are returned to the frontend.

`src/cascade.py` and `src/pan_tompkins.py` are shared, dependency-light modules (no FastAPI/pydantic imports) reused by both the live inference service and the offline evaluation pipeline, so "what the model does in production" and "what got measured" are guaranteed to be the same code path.

Advantages:

* Consistent runtime environment
* Simplified deployment
* Reproducible inference
* Better scalability
* Infrastructure portability

---

# CI/CD Pipeline

The project follows modern software engineering practices using automated CI/CD workflows.

Pipeline stages include:

1. Source code validation
2. Automated testing (`pytest tests/`, including `tests/test_cascade.py` for the shared cascade/detector logic — runs with no Azure credentials or GPU required)
3. Dependency verification
4. Build validation
5. Deployment preparation

Benefits:

* Reduced deployment risk
* Faster feedback cycles
* Improved code quality
* Reproducible releases

---

# Experiment Tracking

Training runs, model artifacts, validation metrics, and candidate models are systematically stored and compared via MLflow (Azure ML-native) to support reproducible experimentation and informed model selection.

The project emphasizes evidence-based model promotion — via the cascade-level evaluation job — rather than relying solely on individual training-run metrics.

---

# Reproducibility

The project was designed with reproducibility as a core principle.

Implemented practices include:

* Dockerized environments
* Version-controlled source code
* Versioned model artifacts
* Persisted per-fold train/val patient splits (`fold_splits_binary.json`, `fold_splits_multiclass.json`) so any past training run's validation partitions can be reconstructed exactly, without re-running the non-deterministic parts of preprocessing
* Automated CI validation
* Standardized evaluation workflows (`--fold`, `--records` flags for cheap smoke-testing before a full run)

---

# Known Limitations

This section is deliberately blunt — these are measured findings from the evaluation snapshot above, not hypothetical caveats.

* **Cascade macro F1 is still moderate (0.34, 95% CI 0.24–0.44).** ~35.4% of true anomalous beats are classified correctly end-to-end, up from ~19.5% before switching the R-peak detector (see [R-Peak Detector Choice](#r-peak-detector-choice)) — but the remaining loss (Stage 1 false-negatives 25.3%, the detector 24.0%, Stage 2 misclassification 15.3%) is still a compounding-error problem across all three stages, not a single fixable bottleneck.
* **Stage 1 sensitivity (0.658) is now the single largest remaining loss category (25.3% of lost anomalies).** Fixing the CV split's stratification (it was blind to AAMI subtype) raised it from 0.577 without trading away specificity or precision — but every anomaly still missed here can never reach Stage 2. Recalibrating the decision threshold away from 0.5 was tried two ways, on two independent training runs, and rejected every time: it either measurably hurts cascade macro F1, or its apparent gain is within bootstrap noise (see Stage 1 above).
* **Multiclass performance is highly unstable across folds** (val accuracy 0.29–0.98, same model/data pipeline), which made every other multiclass experiment unmeasurable. The dominant cause was class Q: 99.8% of it comes from four paced recordings, so each fold's Q validation set was effectively a single patient and fold results swung on *which* pacemaker was held out. Excluding Q (see [AAMI class scope](#aami-class-scope-why-q-paced-beats-are-excluded)) removes that source directly; the residual variance from class concentration in S/V/F is smaller but not eliminated, and has not yet been re-measured.
* **Class F (fusion beats) is not learned at all** — 0.0 precision and recall throughout, both at the window level and in the cascade. Two interventions were measured against a matched baseline on identical fold splits and **both made things worse**: doubling F with WGAN-GP samples (F stayed at 0.00 F1 in 4 of 5 folds), and `balanced` class weighting (F recall rose to 0.06–0.88 but precision collapsed to ~0.00–0.11, i.e. the model was forced to guess F rather than taught to recognise it, while overall val accuracy fell from 0.77 to 0.50 and training destabilised — one fold scored AUC 0.40, below chance). F has ~72–93 training windows per fold against V/Q at 34–43× that; at this data volume it appears genuinely unlearnable rather than under-served.
* **WGAN-GP augmentation shows no measurable benefit**, now tested properly rather than at a no-op dose. The original +5% growth cap generated only 3–4 synthetic F windows per fold; raised to +100% (≈600 synthetic windows per fold, a ~20× increase) and compared against a matched no-GAN run on identical fold splits, the GAN run was **worse on all four aggregate metrics and in 4 of 5 folds** (val accuracy 0.683 vs 0.772). See [WGAN-GP Case Study](#wgan-gp-case-study).
* **The legacy Pan-Tompkins detector degraded badly on atypical records** (0.50–0.55 sensitivity on 2 of 10 tested records vs. ~0.97+ on the rest), which was the largest single source of cascade loss at one point (36.6%) — this motivated switching to `neurokit_detect` (see [R-Peak Detector Choice](#r-peak-detector-choice)), which cut the detector's loss share to 24.0%. Record 108 specifically remains weak (best measured: 0.67 sensitivity, from a 10-algorithm ensemble) across every detector tried so far.
* **Small dataset overall**: `--selected-samples 40` records, which limits both training data volume and how representative each CV fold can be.

---

# Technical Challenges

Key challenges encountered during development:

* ECG class imbalance, especially the near-absence of class F
* Patient-level fold variance dominating multiclass results on a small dataset
* Compounding error across a 3-stage cascade (detection → binary → multiclass)
* Getting a real, end-to-end (not just per-window) evaluation pipeline running reliably on Azure ML, including several live infrastructure bugs (quota limits, job-output URI resolution, mlflow/azureml-mlflow version mismatches, missing dependencies)
* Comparing multiple model candidates fairly under a hard no-service-principal auth constraint
* Managing model versions
* Designing deployment-ready architecture shared between inference and evaluation

---

# Lessons Learned

The project provided practical experience in:

* Cross-validation strategies, and their limits on small, patient-grouped datasets
* Deep Learning model evaluation — and why per-window CV metrics can look fine while the true end-to-end system metric doesn't
* Model lifecycle management
* Cloud-assisted experimentation, including debugging real Azure ML infrastructure issues (quota, job-output referencing, environment/dependency drift)
* Docker-based deployment
* CI/CD integration for ML systems
* Trade-offs between accuracy and generalization
* Production-oriented ML development

---

# Potential Use Cases

Potential applications include:

* Clinical decision support systems
* ECG screening tools
* Telemedicine platforms
* Research environments
* Medical AI prototyping
* Healthcare analytics solutions

(Given the current [Known Limitations](#known-limitations), the system today is a research/portfolio-stage prototype, not something ready for any of the above without further work on Stage 1 sensitivity and fold stability.)

---

# Roadmap

## Completed

* Binary classification model
* Multiclass classification model
* Cross-validation workflow
* Azure-based training (both stages, one job)
* WGAN-GP minority-class augmentation, evaluated fold-by-fold against a no-GAN baseline
* End-to-end cascade evaluation pipeline on Azure ML (Pan-Tompkins + both stages, baselines, CPU benchmarks, in-job promotion decision)
* Dockerized inference
* REST API integration
* Streamlit frontend
* CI/CD implementation
* Champion vs Challenger evaluation (now computed in-job, not locally)
* Model versioning
* Switched the cascade's R-peak detector from a custom Pan-Tompkins implementation to `neurokit2` after a multi-stage, statistically validated comparison — see [R-Peak Detector Choice](#r-peak-detector-choice)
* Settled the WGAN-GP question with a matched-baseline experiment at a meaningful augmentation dose, rather than leaving it "inconclusive" — see [Known Limitations](#known-limitations)

## In Progress

* **Excluding class Q (paced beats) from both stages** — decision made and documented in [AAMI class scope](#aami-class-scope-why-q-paced-beats-are-excluded); implementation and the first post-change training/evaluation runs are the immediate next step. Everything below is blocked on that, because Q dominated the fold variance that made other effects unmeasurable.
* Re-measuring the whole evaluation suite on the S/V/F label set — every number in [Model Evaluation](#model-evaluation) predates the exclusion and will move
* Improving Stage 1 sensitivity further (0.577 → 0.658 via CV stratification fix; the largest remaining loss category at 25.3% of lost anomalies). Threshold recalibration away from 0.5 was tried and rejected twice — see Stage 1 above.
* Record 108 remains weak across every R-peak detector tried so far (best: 0.67 sensitivity, from a 10-algorithm ensemble) — may need per-record adaptive thresholding rather than a better fixed algorithm
* Deciding what to do about class F, which neither synthetic augmentation nor class weighting made learnable — the realistic options are more data, merging it, or reporting it as out of scope

---

# Skills Demonstrated

This project showcases practical competencies relevant to Machine Learning Engineering roles.

## Machine Learning

* Deep Learning
* Classification Models
* TensorFlow/Keras
* Cross Validation
* Model Evaluation (including honest end-to-end system evaluation, not just per-window metrics)
* Hyperparameter Experimentation

## MLOps

* CI/CD
* Docker
* Model Versioning
* Experiment Tracking
* Champion vs Challenger Strategy, designed around a real cloud-auth constraint

## Cloud Engineering

* Microsoft Azure ML
* Training and evaluation infrastructure
* Artifact management
* Live debugging of cloud quota, credential, and dependency issues

## Software Engineering

* REST APIs
* Modular Architecture (shared cascade logic between service and evaluation pipeline)
* Deployment Workflows
* Reproducible Systems
* Version Control

## AI Engineering

* Generative AI Integration (WGAN-GP)
* Claude Code / Gemini Code Assist
* ML System Design

# ECG Report AI

## Overview

ECG Report AI is an end-to-end Machine Learning project focused on automated ECG (Electrocardiogram) classification using Deep Learning techniques.

The project was built to demonstrate practical Machine Learning Engineering skills across the complete ML lifecycle, including model development, experimentation, validation, cloud-assisted training, containerized deployment, CI/CD automation, and model lifecycle management.

Rather than focusing solely on model performance, the project emphasizes reproducibility, maintainability, deployment readiness, and industry-standard MLOps practices — including honestly reporting what the system does and does not yet do well (see [Known Limitations](#known-limitations)).

The system analyzes ECG recordings from the MIT-BIH Arrhythmia Database and serves predictions through a Streamlit application connected to a containerized inference API.

> **Evaluation snapshot: 2026-08-10.** All numbers below come from a real evaluation job run against the live cascade (Pan-Tompkins → binary → multiclass) on held-out MIT-BIH patients — not training-time metrics. Re-running the evaluation pipeline against a newer training job will produce different numbers; this README reflects the most recent run only.

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
* Multiclass ECG classification (AAMI classes S/V/F/Q)
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
                  (AAMI: S / V / F / Q)
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

This is the number that matters most: running the *actual* production pipeline (Pan-Tompkins detection → Stage 1 binary → Stage 2 multiclass) on 9 held-out full ECG records, matching detected peaks back to ground-truth annotations (±5 samples).

| Metric | Value |
| --- | --- |
| **Cascade macro F1** | **0.2410** |
| Cascade weighted F1 | 0.7609 |
| Cascade accuracy | 0.7285 |
| Records evaluated | 9 |

Per-class breakdown at the cascade output:

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| N | 0.9054 | 0.8270 | 0.8644 | 15,915 |
| S | 0.0134 | 0.0962 | 0.0235 | 52 |
| V | 0.0504 | 0.1725 | 0.0780 | 742 |
| F | 0.0000 | 0.0000 | 0.0000 | 18 |
| Q | 0.3044 | 0.1968 | 0.2391 | 2,088 |

**Where true anomalous beats are lost** (2,900 true anomalies total across the 9 records):

| Outcome | Count | Share |
| --- | --- | --- |
| Missed by the R-peak detector | 601 | 20.7% |
| Detected, but classified Normal by Stage 1 | 774 | 26.7% |
| Reached Stage 2, but misclassified | 981 | 33.8% |
| Correctly classified end-to-end | 544 | **18.8%** |

Only about 1 in 5 true anomalous beats survives the full cascade correctly, and the losses are spread fairly evenly across all three stages — this is not a single bottleneck, it's a compounding-error problem (see [Known Limitations](#known-limitations)).

![Confusion matrix](docs/img/confusion_matrix.png)

---

## Stage 1 — Binary (Normal vs. Anomaly)

Clinical metrics, pooled across all 5 CV folds:

| Metric | Value |
| --- | --- |
| Sensitivity (recall on anomalies) | **0.5768** |
| Specificity | 0.8951 |
| PPV | 0.4476 |
| NPV | 0.9349 |

Stage 1 sensitivity is the ceiling on the whole cascade: it misses ~42% of true anomalies before they ever reach Stage 2, which alone accounts for more end-to-end loss than Stage 2's own misclassifications.

## Stage 2 — Multiclass (AAMI: S / V / F / Q)

Aggregate classification report across all 5 folds (window-level, not the cascade — see above for the end-to-end number):

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| S | 0.5338 | 0.3410 | 0.4162 | 648 |
| V | 0.4477 | 0.8651 | 0.5900 | 4,765 |
| F | 0.0000 | 0.0000 | 0.0000 | 103 |
| Q | 0.8363 | 0.4093 | 0.5496 | 8,041 |

| Metric | Value |
| --- | --- |
| Macro F1 | 0.3889 |
| Weighted F1 | 0.5532 |
| Accuracy | 0.5631 |
| Mean CV AUC | 0.7819 |

**Per-fold breakdown — variance is the dominant story here:**

| Fold | Val Acc | Val AUC | Macro F1 | Weighted F1 |
| --- | --- | --- | --- | --- |
| 0 | 0.4388 | 0.5051 | 0.2851 | 0.4713 |
| 1 | 0.9760 | 0.9922 | 0.6365 | 0.9716 |
| 2 | 0.2166 | 0.6497 | 0.1086 | 0.1217 |
| 3 | 0.8711 | 0.7827 | 0.3415 | 0.8719 |
| 4 | 0.9739 | 0.9797 | 0.5010 | 0.9817 |

Val accuracy ranges from 0.22 to 0.98 across 5 folds of the *same* model and pipeline. Against majority-class and stratified-random baselines, the model clearly beats both in 4 of 5 folds — but **loses to both baselines in fold 2** (model macro F1 0.109 vs. baseline 0.218/0.226), which the aggregate numbers above would otherwise hide.

![Per-fold metrics](docs/img/per_fold_metrics.png)

---

## Pan-Tompkins Detector Validation

Extended from an earlier 2-record spot check to 10 MIT-BIH records, tolerance ±5 samples:

| Metric | Value |
| --- | --- |
| Aggregate sensitivity | 0.8840 |
| Aggregate PPV | 0.9556 |

Per-record, most are ≥0.97 sensitivity, but two records are notably worse — record 104 (0.5464) and record 108 (0.5026) — both known in the MIT-BIH literature for atypical/noisy morphology, suggesting the detector's fixed adaptive-threshold parameters don't generalize to every recording condition.

![Pan-Tompkins detection](docs/img/pan_tompkins_detection.png)

---

## CPU Inference Benchmarks

Measured on the training/evaluation compute (`Standard_DS3_v2`, 4 vCPU, CPU-only):

| Metric | Value |
| --- | --- |
| Mean Pan-Tompkins detection time | 0.039 s/record |
| Mean Stage 1 (binary) inference time | 1.372 s/record |
| Mean Stage 2 (multiclass) inference time | 0.644 s/record |
| Per-beat latency (mean) | 1.03 ms |
| Per-beat latency (p95) | 1.37 ms |
| **Batched vs. per-window `predict()` speedup** | **~84x** |
| Binary model parameters | 438,081 |
| Multiclass model parameters | 438,468 |
| Binary model size | 5.4 MB |
| Multiclass model size | 5.4 MB |

The ~84x batching speedup quantifies the payoff of refactoring `run_stream` from sequential to batched inference in the FastAPI service.

---

## WGAN-GP Case Study

WGAN-GP (Conv1DTranspose generator/critic, `d_steps=3`, gradient penalty) is used to top up rare AAMI classes (S, F, sometimes V) during multiclass training, capped at +5% growth per class per fold to avoid overwhelming real data with synthetic samples.

A matching pair of training runs — one with GAN augmentation, one with `--skip-gan-augmentation` — were evaluated fold-by-fold:

| Fold | Val Acc (GAN) | Val Acc (no-GAN) | Difference |
| --- | --- | --- | --- |
| 0 | 0.4388 | 0.4889 | -0.0500 |
| 1 | 0.9760 | 0.9736 | +0.0023 |
| 2 | 0.2166 | 0.3199 | -0.1033 |
| 3 | 0.8711 | 0.7967 | +0.0744 |
| 4 | 0.9739 | 0.9501 | +0.0238 |

**Verdict: inconclusive.** Mean Val Acc difference across folds is -0.0106, well within the ±0.347 inter-fold standard deviation — the GAN's effect, whatever it is, is currently smaller than the noise already present from patient-level fold variance. This is a real, measured result, not a placeholder: at the current augmentation strength and dataset size, WGAN-GP augmentation has not been shown to move the needle.

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

* **Cascade macro F1 is low (0.24).** Only ~19% of true anomalous beats are classified correctly end-to-end; the rest are lost roughly evenly across detection (21%), Stage 1 false-negatives (27%), and Stage 2 misclassification (34%). This is a compounding-error problem across all three stages, not a single fixable bottleneck.
* **Stage 1 sensitivity (0.577) is the practical ceiling on the whole system.** Improving it would have the largest single effect on cascade performance, since anomalies missed here can never reach Stage 2.
* **Multiclass performance is highly unstable across folds** (val accuracy 0.22–0.98, same model/data pipeline). Fold 2 in particular has the model *losing* to a trivial majority-class baseline. With only 26–31 training patients per fold, individual patients' morphology dominates fold-to-fold results — this is a dataset-size/patient-heterogeneity problem, not primarily a modeling one.
* **Class F (fusion beats) is not learned at all** — 0.0 precision and recall throughout, both at the window level and in the cascade. F has the fewest samples of any AAMI class (18 in the cascade evaluation set); the current data volume can't support learning it.
* **WGAN-GP augmentation shows no measurable benefit** at current settings (mean Val Acc difference -0.011, within a ±0.347 inter-fold noise band) — see [WGAN-GP Case Study](#wgan-gp-case-study).
* **The Pan-Tompkins detector degrades on atypical records** (0.50–0.55 sensitivity on 2 of 10 tested records vs. ~0.97+ on the rest) — its fixed thresholding doesn't generalize to every signal condition in MIT-BIH.
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

## In Progress

* Improving Stage 1 sensitivity (currently the main cascade bottleneck)
* Reducing multiclass fold-to-fold variance (likely needs more patients per fold, not just more augmentation)
* Making WGAN-GP augmentation actually move the needle, or concluding it isn't the right lever for this dataset size
* Pan-Tompkins detector robustness on atypical records

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

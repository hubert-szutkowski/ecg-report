# fastapi_ecg_service/main.py

import logging
import tempfile
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
import wfdb
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from src.AAMI_classification_multistages.multistage_model import PositionalEmbedding
from .pan_tompkins import pan_tompkins_detect

WEIGHTS_DIR = Path("/app/weights")
BINARY_PATH = WEIGHTS_DIR / "ecg_binary_model.keras"
MULTICLASS_PATH = WEIGHTS_DIR / "ecg_multiclass_model.keras"
SCALER_PATH = WEIGHTS_DIR / "scaler.pkl"

WINDOW_SIZE = 216
HALF_WINDOW = WINDOW_SIZE // 2
BINARY_THRESHOLD = 0.5

CUSTOM_OBJECTS = {"PositionalEmbedding": PositionalEmbedding}

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("ecg_service")

app = FastAPI(
    title="ECG Inception-Conformer Classifier",
    description="Two-stage ECG classification from uploaded WFDB .hea/.dat files, with Pan-Tompkins R-peak detection.",
    version="3.0.0",
)

binary_model: tf.keras.Model | None = None
multiclass_model: tf.keras.Model | None = None
scaler = None


@app.on_event("startup")
def load_models() -> None:
    global binary_model, multiclass_model, scaler

    if not BINARY_PATH.exists():
        raise RuntimeError(f"Binary model weights not found: {BINARY_PATH}")
    if not SCALER_PATH.exists():
        raise RuntimeError(f"Scaler file not found: {SCALER_PATH}")

    logger.info("Loading binary model from %s", BINARY_PATH)
    binary_model = tf.keras.models.load_model(str(BINARY_PATH), custom_objects=CUSTOM_OBJECTS)

    if MULTICLASS_PATH.exists():
        logger.info("Loading multiclass model from %s", MULTICLASS_PATH)
        multiclass_model = tf.keras.models.load_model(str(MULTICLASS_PATH), custom_objects=CUSTOM_OBJECTS)
    else:
        logger.warning("Multiclass weights not found (%s) — cascade stage 2 disabled.", MULTICLASS_PATH)
        multiclass_model = None

    logger.info("Loading scaler from %s", SCALER_PATH)
    scaler = joblib.load(SCALER_PATH)

    expected_features = getattr(scaler, "n_features_in_", None)
    if expected_features is not None and expected_features != WINDOW_SIZE:
        raise RuntimeError(
            f"Scaler expects {expected_features} features but WINDOW_SIZE={WINDOW_SIZE} — wrong scaler file?"
        )

    logger.info("Startup complete.")


# --- Windowing (must match training exactly) ---

def extract_windows_around_peaks(signal: np.ndarray, peak_indices: np.ndarray, window_size: int = WINDOW_SIZE):
    half_window = window_size // 2
    n_samples = len(signal)
    windows = []
    valid_peaks = []
    for peak in peak_indices:
        start = int(peak) - half_window
        end = start + window_size
        if start >= 0 and end <= n_samples:
            windows.append(signal[start:end])
            valid_peaks.append(int(peak))
    if not windows:
        return np.empty((0, window_size), dtype=np.float32), np.empty((0,), dtype=int)
    return np.stack(windows).astype(np.float32), np.array(valid_peaks)


def normalize_windows(windows_raw: np.ndarray) -> np.ndarray:
    return scaler.transform(windows_raw).astype(np.float32)


def run_cascade_batch(signal: np.ndarray, fs: float) -> "BeatStreamResponse":
    if not np.all(np.isfinite(signal)):
        raise HTTPException(status_code=400, detail="Signal contains NaN or infinite values.")

    peak_indices = pan_tompkins_detect(signal, fs)
    logger.info("Pan-Tompkins detected %d candidate peaks", len(peak_indices))

    windows_raw, valid_peaks = extract_windows_around_peaks(signal, peak_indices)
    if len(valid_peaks) == 0:
        raise HTTPException(status_code=400, detail="No valid beats found (peaks too close to signal boundaries).")

    try:
        windows_normalized = normalize_windows(windows_raw)
    except Exception as exc:
        logger.exception("Scaler normalization failed")
        raise HTTPException(status_code=500, detail="Internal error during signal normalization.") from exc

    x_all = windows_normalized.reshape(-1, WINDOW_SIZE, 1)

    try:
        binary_probs = binary_model.predict(x_all, batch_size=256, verbose=0).flatten()
    except Exception as exc:
        logger.exception("Batched binary inference failed")
        raise HTTPException(status_code=500, detail="Internal error during binary prediction.") from exc

    is_anomaly_mask = binary_probs >= BINARY_THRESHOLD

    multiclass_probs_by_index: dict[int, list[float]] = {}
    predicted_class_by_index: dict[int, int] = {}

    if multiclass_model is not None and is_anomaly_mask.any():
        anomaly_indices = np.where(is_anomaly_mask)[0]
        try:
            class_probs_batch = multiclass_model.predict(x_all[anomaly_indices], batch_size=256, verbose=0)
        except Exception as exc:
            logger.exception("Batched multiclass inference failed")
            raise HTTPException(status_code=500, detail="Internal error during multiclass prediction.") from exc

        for i, idx in enumerate(anomaly_indices):
            multiclass_probs_by_index[int(idx)] = class_probs_batch[i].tolist()
            predicted_class_by_index[int(idx)] = int(np.argmax(class_probs_batch[i]))

    results: list[BeatResult] = []
    for i, peak in enumerate(valid_peaks):
        results.append(
            BeatResult(
                beat_index=i,
                peak_sample=int(peak),
                binary_anomaly_probability=float(binary_probs[i]),
                is_anomaly=bool(is_anomaly_mask[i]),
                multiclass_probabilities=multiclass_probs_by_index.get(i),
                predicted_class=predicted_class_by_index.get(i),
            )
        )

    logger.info("Cascade complete: %d beats, %d anomalies", len(results), int(is_anomaly_mask.sum()))
    return BeatStreamResponse(
        total_beats=len(results),
        detected_peaks=len(peak_indices),
        window_size=WINDOW_SIZE,
        fs=fs,
        results=results,
    )


def load_from_bytes(hea_bytes: bytes, dat_bytes: bytes, base_name: str, channel: int = 0):
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        (tmp_path / f"{base_name}.hea").write_bytes(hea_bytes)
        (tmp_path / f"{base_name}.dat").write_bytes(dat_bytes)

        record = wfdb.rdrecord(str(tmp_path / base_name))
        signal = record.p_signal[:, channel].astype(np.float32)
        return signal, record.fs


# --- Schemas ---

class BeatResult(BaseModel):
    beat_index: int
    peak_sample: int
    binary_anomaly_probability: float
    is_anomaly: bool
    multiclass_probabilities: list[float] | None = None
    predicted_class: int | None = None


class BeatStreamResponse(BaseModel):
    total_beats: int
    detected_peaks: int
    window_size: int
    fs: float
    results: list[BeatResult]


class HealthResponse(BaseModel):
    status: str
    binary_loaded: bool
    multiclass_loaded: bool
    scaler_loaded: bool


# --- Endpoints ---

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if binary_model is not None and scaler is not None else "degraded",
        binary_loaded=binary_model is not None,
        multiclass_loaded=multiclass_model is not None,
        scaler_loaded=scaler is not None,
    )


@app.post("/predict_from_upload", response_model=BeatStreamResponse)
async def predict_from_upload(
    hea_file: UploadFile = File(...),
    dat_file: UploadFile = File(...),
    channel: int = Form(0),
) -> BeatStreamResponse:
    if binary_model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model or scaler not loaded.")

    base_name = Path(hea_file.filename).stem
    try:
        hea_bytes = await hea_file.read()
        dat_bytes = await dat_file.read()
        signal, fs = load_from_bytes(hea_bytes, dat_bytes, base_name, channel)
    except Exception as exc:
        logger.exception("Failed to load uploaded record")
        raise HTTPException(status_code=400, detail=f"Could not load uploaded files: {exc}") from exc

    return run_cascade_batch(signal, fs)
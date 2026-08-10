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
from src.cascade import run_cascade_batch as run_cascade_core

WEIGHTS_DIR = Path("/app/weights")
BINARY_PATH = WEIGHTS_DIR / "ecg_binary_model.keras"
MULTICLASS_PATH = WEIGHTS_DIR / "ecg_multiclass_model.keras"
SCALER_PATH = WEIGHTS_DIR / "scaler.pkl"

WINDOW_SIZE = 216
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


# --- Cascade (windowing/detection/inference logic lives in src/cascade.py) ---

def run_cascade_batch(signal: np.ndarray, fs: float) -> "BeatStreamResponse":
    if not np.all(np.isfinite(signal)):
        raise HTTPException(status_code=400, detail="Signal contains NaN or infinite values.")

    try:
        cascade_result = run_cascade_core(
            signal,
            fs,
            binary_model=binary_model,
            scaler=scaler,
            multiclass_model=multiclass_model,
            window_size=WINDOW_SIZE,
            binary_threshold=BINARY_THRESHOLD,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.exception("Cascade inference failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    results = [BeatResult(**beat) for beat in cascade_result["results"]]
    n_anomalies = sum(1 for beat in cascade_result["results"] if beat["is_anomaly"])
    logger.info(
        "Pan-Tompkins detected %d candidate peaks", cascade_result["detected_peaks"]
    )
    logger.info("Cascade complete: %d beats, %d anomalies", len(results), n_anomalies)

    return BeatStreamResponse(
        total_beats=cascade_result["total_beats"],
        detected_peaks=cascade_result["detected_peaks"],
        window_size=cascade_result["window_size"],
        fs=cascade_result["fs"],
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
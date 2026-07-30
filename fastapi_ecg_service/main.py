# fastapi_ecg_service/main.py
import joblib
import logging
import pickle
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import requests
import tensorflow as tf
import wfdb
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field, field_validator

from src.AAMI_classification_multistages.multistage_model import PositionalEmbedding

WEIGHTS_DIR = Path("/app/weights")
BINARY_PATH = WEIGHTS_DIR / "ecg_binary_model.keras"
MULTICLASS_PATH = WEIGHTS_DIR / "ecg_multiclass_model.keras"
SCALER_PATH = WEIGHTS_DIR / "scaler.pkl"

WINDOW_SIZE = 216
DEFAULT_STRIDE = 216
BINARY_THRESHOLD = 0.5

CUSTOM_OBJECTS = {"PositionalEmbedding": PositionalEmbedding}

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("ecg_service")

app = FastAPI(
    title="ECG Inception-Conformer Classifier",
    description="Two-stage ECG classification: binary anomaly detection + multiclass cascade.",
    version="1.4.0",
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


# --- Normalization + inference ---

def normalize_windows(windows_raw: np.ndarray) -> np.ndarray:
    # windows_raw: shape (n_windows, WINDOW_SIZE) — scaler.transform, never fit
    return scaler.transform(windows_raw).astype(np.float32)


def run_cascade(window_raw: np.ndarray) -> dict:
    normalized = normalize_windows(window_raw.reshape(1, WINDOW_SIZE))
    x = normalized.reshape(1, WINDOW_SIZE, 1)

    binary_prob = float(binary_model.predict(x, verbose=0)[0][0])
    is_anomaly = binary_prob >= BINARY_THRESHOLD

    result = {"binary_anomaly_probability": binary_prob, "is_anomaly": is_anomaly}
    if is_anomaly and multiclass_model is not None:
        class_probs = multiclass_model.predict(x, verbose=0)[0]
        result["multiclass_probabilities"] = class_probs.tolist()
        result["predicted_class"] = int(np.argmax(class_probs))
    return result


def run_stream(signal_raw: np.ndarray, stride: int) -> "StreamResponse":
    if len(signal_raw) < WINDOW_SIZE:
        raise HTTPException(status_code=400, detail=f"Signal must have at least {WINDOW_SIZE} samples.")
    if not np.all(np.isfinite(signal_raw)):
        raise HTTPException(status_code=400, detail="Signal contains NaN or infinite values.")

    starts = list(range(0, len(signal_raw) - WINDOW_SIZE + 1, stride))
    windows_raw = np.stack([signal_raw[s: s + WINDOW_SIZE] for s in starts])

    try:
        windows_normalized = normalize_windows(windows_raw)
    except Exception as exc:
        logger.exception("Scaler normalization failed")
        raise HTTPException(status_code=500, detail="Internal error during signal normalization.") from exc

    x_all = windows_normalized.reshape(-1, WINDOW_SIZE, 1)
    n_windows = len(starts)

    logger.info("Running binary model on %d windows (batched)", n_windows)
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
        logger.info("Running multiclass model on %d anomalous windows (batched)", len(anomaly_indices))
        try:
            x_anomaly = x_all[anomaly_indices]
            class_probs_batch = multiclass_model.predict(x_anomaly, batch_size=256, verbose=0)
        except Exception as exc:
            logger.exception("Batched multiclass inference failed")
            raise HTTPException(status_code=500, detail="Internal error during multiclass prediction.") from exc

        for i, idx in enumerate(anomaly_indices):
            probs = class_probs_batch[i].tolist()
            multiclass_probs_by_index[int(idx)] = probs
            predicted_class_by_index[int(idx)] = int(np.argmax(class_probs_batch[i]))

    results: list[WindowResult] = []
    for window_index, start in enumerate(starts):
        results.append(
            WindowResult(
                window_index=window_index,
                start_sample=start,
                binary_anomaly_probability=float(binary_probs[window_index]),
                is_anomaly=bool(is_anomaly_mask[window_index]),
                multiclass_probabilities=multiclass_probs_by_index.get(window_index),
                predicted_class=predicted_class_by_index.get(window_index),
            )
        )

    logger.info("Stream complete: %d windows, %d anomalies", n_windows, int(is_anomaly_mask.sum()))
    return StreamResponse(total_windows=n_windows, window_size=WINDOW_SIZE, stride=stride, results=results)


# --- wfdb loaders ---

def load_from_physionet(record_name: str, pn_dir: str, channel: int = 0) -> np.ndarray:
    record = wfdb.rdrecord(record_name, pn_dir=pn_dir)
    return record.p_signal[:, channel].astype(np.float32)


def load_from_url(hea_url: str, dat_url: str, channel: int = 0) -> np.ndarray:
    base_name = Path(urlparse(hea_url).path).stem
    if Path(urlparse(dat_url).path).stem != base_name:
        raise ValueError(".hea and .dat file names must match (same record).")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        for url, suffix in [(hea_url, ".hea"), (dat_url, ".dat")]:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            (tmp_path / f"{base_name}{suffix}").write_bytes(resp.content)

        record = wfdb.rdrecord(str(tmp_path / base_name))
        return record.p_signal[:, channel].astype(np.float32)


def load_from_bytes(hea_bytes: bytes, dat_bytes: bytes, base_name: str, channel: int = 0) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        (tmp_path / f"{base_name}.hea").write_bytes(hea_bytes)
        (tmp_path / f"{base_name}.dat").write_bytes(dat_bytes)

        record = wfdb.rdrecord(str(tmp_path / base_name))
        return record.p_signal[:, channel].astype(np.float32)


# --- Schemas ---

class PredictRequest(BaseModel):
    signal: list[float] = Field(..., min_length=WINDOW_SIZE, max_length=WINDOW_SIZE)

    @field_validator("signal")
    @classmethod
    def validate_finite(cls, value: list[float]) -> list[float]:
        arr = np.asarray(value, dtype=np.float64)
        if not np.all(np.isfinite(arr)):
            raise ValueError("Signal contains NaN or infinite values.")
        return value


class SignalStreamRequest(BaseModel):
    signal: list[float] = Field(..., min_length=WINDOW_SIZE)
    stride: int = Field(default=DEFAULT_STRIDE, ge=1, le=WINDOW_SIZE)

    @field_validator("signal")
    @classmethod
    def validate_finite(cls, value: list[float]) -> list[float]:
        arr = np.asarray(value, dtype=np.float64)
        if not np.all(np.isfinite(arr)):
            raise ValueError("Signal contains NaN or infinite values.")
        return value


class PhysionetRequest(BaseModel):
    record_name: str = Field(..., description="e.g. '100' for mitdb/100")
    pn_dir: str = Field(..., description="e.g. 'mitdb'")
    channel: int = Field(default=0, ge=0)
    stride: int = Field(default=DEFAULT_STRIDE, ge=1, le=WINDOW_SIZE)


class UrlSignalRequest(BaseModel):
    hea_url: str
    dat_url: str
    channel: int = Field(default=0, ge=0)
    stride: int = Field(default=DEFAULT_STRIDE, ge=1, le=WINDOW_SIZE)


class WindowResult(BaseModel):
    window_index: int
    start_sample: int
    binary_anomaly_probability: float
    is_anomaly: bool
    multiclass_probabilities: list[float] | None = None
    predicted_class: int | None = None


class StreamResponse(BaseModel):
    total_windows: int
    window_size: int
    stride: int
    results: list[WindowResult]


class PredictResponse(BaseModel):
    binary_anomaly_probability: float
    is_anomaly: bool
    multiclass_probabilities: list[float] | None = None
    predicted_class: int | None = None


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


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    if binary_model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model or scaler not loaded.")

    window = np.asarray(request.signal, dtype=np.float32)
    try:
        result = run_cascade(window)
    except Exception as exc:
        logger.exception("Inference failed in /predict")
        raise HTTPException(status_code=500, detail="Internal error during prediction.") from exc

    return PredictResponse(**result)


@app.post("/predict_stream", response_model=StreamResponse)
def predict_stream(request: SignalStreamRequest) -> StreamResponse:
    if binary_model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model or scaler not loaded.")
    signal = np.asarray(request.signal, dtype=np.float32)
    return run_stream(signal, request.stride)


@app.post("/predict_from_physionet", response_model=StreamResponse)
def predict_from_physionet(request: PhysionetRequest) -> StreamResponse:
    if binary_model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model or scaler not loaded.")

    try:
        signal = load_from_physionet(request.record_name, request.pn_dir, request.channel)
    except Exception as exc:
        logger.exception("Failed to fetch record from PhysioNet")
        raise HTTPException(status_code=400, detail=f"Could not fetch record: {exc}") from exc

    return run_stream(signal, request.stride)


@app.post("/predict_from_url", response_model=StreamResponse)
def predict_from_url(request: UrlSignalRequest) -> StreamResponse:
    if binary_model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model or scaler not loaded.")

    try:
        signal = load_from_url(request.hea_url, request.dat_url, request.channel)
    except Exception as exc:
        logger.exception("Failed to fetch/load signal from URL")
        raise HTTPException(status_code=400, detail=f"Could not fetch/load signal: {exc}") from exc

    return run_stream(signal, request.stride)


@app.post("/predict_from_upload", response_model=StreamResponse)
async def predict_from_upload(
    hea_file: UploadFile = File(...),
    dat_file: UploadFile = File(...),
    channel: int = Form(0),
    stride: int = Form(DEFAULT_STRIDE),
) -> StreamResponse:
    if binary_model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model or scaler not loaded.")

    base_name = Path(hea_file.filename).stem
    try:
        hea_bytes = await hea_file.read()
        dat_bytes = await dat_file.read()
        signal = load_from_bytes(hea_bytes, dat_bytes, base_name, channel)
    except Exception as exc:
        logger.exception("Failed to load uploaded record")
        raise HTTPException(status_code=400, detail=f"Could not load uploaded files: {exc}") from exc

    return run_stream(signal, stride)
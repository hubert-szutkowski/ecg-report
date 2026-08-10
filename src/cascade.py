import numpy as np

# src/ is imported flat (Azure ML jobs, PYTHONPATH=.) and as the "src" package
# (fastapi_ecg_service, which does `from src.cascade import ...`) - support both.
try:
    from pan_tompkins import pan_tompkins_detect
except ImportError:
    from src.pan_tompkins import pan_tompkins_detect


def extract_windows_around_peaks(signal: np.ndarray, peak_indices: np.ndarray, window_size: int = 216):
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


def normalize_windows(windows_raw: np.ndarray, scaler) -> np.ndarray:
    return scaler.transform(windows_raw).astype(np.float32)


def run_cascade_batch(
    signal: np.ndarray,
    fs: float,
    binary_model,
    scaler,
    multiclass_model=None,
    window_size: int = 216,
    binary_threshold: float = 0.5,
    batch_size: int = 256,
) -> dict:
    if not np.all(np.isfinite(signal)):
        raise ValueError("Signal contains NaN or infinite values.")

    peak_indices = pan_tompkins_detect(signal, fs)
    windows_raw, valid_peaks = extract_windows_around_peaks(signal, peak_indices, window_size)
    if len(valid_peaks) == 0:
        raise ValueError("No valid beats found (peaks too close to signal boundaries).")

    try:
        windows_normalized = normalize_windows(windows_raw, scaler)
    except Exception as exc:
        raise RuntimeError("Internal error during signal normalization.") from exc

    x_all = windows_normalized.reshape(-1, window_size, 1)

    try:
        binary_probs = binary_model.predict(x_all, batch_size=batch_size, verbose=0).flatten()
    except Exception as exc:
        raise RuntimeError("Internal error during binary prediction.") from exc

    is_anomaly_mask = binary_probs >= binary_threshold

    multiclass_probs_by_index: dict[int, list[float]] = {}
    predicted_class_by_index: dict[int, int] = {}

    if multiclass_model is not None and is_anomaly_mask.any():
        anomaly_indices = np.where(is_anomaly_mask)[0]
        try:
            class_probs_batch = multiclass_model.predict(x_all[anomaly_indices], batch_size=batch_size, verbose=0)
        except Exception as exc:
            raise RuntimeError("Internal error during multiclass prediction.") from exc

        for i, idx in enumerate(anomaly_indices):
            multiclass_probs_by_index[int(idx)] = class_probs_batch[i].tolist()
            predicted_class_by_index[int(idx)] = int(np.argmax(class_probs_batch[i]))

    results = []
    for i, peak in enumerate(valid_peaks):
        results.append({
            "beat_index": i,
            "peak_sample": int(peak),
            "binary_anomaly_probability": float(binary_probs[i]),
            "is_anomaly": bool(is_anomaly_mask[i]),
            "multiclass_probabilities": multiclass_probs_by_index.get(i),
            "predicted_class": predicted_class_by_index.get(i),
        })

    return {
        "total_beats": len(results),
        "detected_peaks": len(peak_indices),
        "window_size": window_size,
        "fs": fs,
        "results": results,
    }


def match_detected_to_annotated_peaks(
    detected_peaks: np.ndarray,
    annotated_samples: np.ndarray,
    tolerance_samples: int = 5,
) -> dict:
    """
    Parameters:
        - detected_peaks: np.ndarray of Pan-Tompkins detected peak sample indices
        - annotated_samples: np.ndarray of ground-truth annotated peak sample indices,
          pre-filtered to beats with a known AAMI class (see SYMBOL_TO_CLASS) - non-beat
          annotation symbols must not be passed in, or they are counted as missed beats
        - tolerance_samples: max sample distance for a detected/annotated pair to match
    Returns:
        - dict with matched_annotated_idx, matched_detected_idx (parallel index arrays
          into the two inputs), unmatched_annotated_idx (missed by the detector),
          unmatched_detected_idx (spurious detections)
    """
    detected_peaks = np.asarray(detected_peaks)
    annotated_samples = np.asarray(annotated_samples)

    used_detected = np.zeros(len(detected_peaks), dtype=bool)
    matched_annotated_idx = []
    matched_detected_idx = []

    for a_idx, a_sample in enumerate(annotated_samples):
        distances = np.abs(detected_peaks - a_sample)
        candidates = np.where(~used_detected & (distances <= tolerance_samples))[0]
        if len(candidates) == 0:
            continue
        best = candidates[np.argmin(distances[candidates])]
        used_detected[best] = True
        matched_annotated_idx.append(a_idx)
        matched_detected_idx.append(best)

    matched_annotated_idx = np.array(matched_annotated_idx, dtype=int)
    matched_detected_idx = np.array(matched_detected_idx, dtype=int)

    return {
        "matched_annotated_idx": matched_annotated_idx,
        "matched_detected_idx": matched_detected_idx,
        "unmatched_annotated_idx": np.setdiff1d(np.arange(len(annotated_samples)), matched_annotated_idx),
        "unmatched_detected_idx": np.setdiff1d(np.arange(len(detected_peaks)), matched_detected_idx),
    }

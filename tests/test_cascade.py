import os
import sys

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import cascade


def test_extract_windows_around_peaks_filters_boundary_peaks():
    """
    Ensure windows are sliced exactly as start = peak - window_size//2, and that
    peaks too close to either signal boundary are dropped rather than truncated.
    """
    signal = np.arange(1000, dtype=np.float32)
    peak_indices = [3, 50, 998]
    window_size = 10

    windows, valid_peaks = cascade.extract_windows_around_peaks(signal, peak_indices, window_size)

    assert valid_peaks.tolist() == [50], f"Expected only peak 50 to survive, got {valid_peaks.tolist()}"
    assert windows.shape == (1, 10), f"Expected shape (1, 10), got {windows.shape}"
    np.testing.assert_array_equal(windows[0], signal[45:55])


def test_extract_windows_around_peaks_empty_when_no_valid_peaks():
    signal = np.arange(20, dtype=np.float32)
    windows, valid_peaks = cascade.extract_windows_around_peaks(signal, [0, 19], window_size=10)

    assert len(windows) == 0
    assert len(valid_peaks) == 0


def test_normalize_windows_uses_passed_scaler_transform_only():
    """
    normalize_windows must call .transform() on the scaler it's given and never
    refit it - a fold's persisted scaler must never see validation/inference data
    through .fit() or .fit_transform().
    """
    rng = np.random.default_rng(0)
    scaler = StandardScaler().fit(rng.normal(size=(50, 10)))
    mean_before = scaler.mean_.copy()

    windows_raw = rng.normal(size=(4, 10)).astype(np.float32)
    normalized = cascade.normalize_windows(windows_raw, scaler)

    np.testing.assert_array_almost_equal(scaler.mean_, mean_before)
    np.testing.assert_array_almost_equal(normalized, scaler.transform(windows_raw).astype(np.float32))


def test_match_detected_to_annotated_peaks_within_tolerance():
    """
    A detected peak within tolerance of an annotated peak counts as a match;
    a detected peak that matches nothing is a spurious detection.
    """
    detected = np.array([100, 205, 300])
    annotated = np.array([102, 300])

    result = cascade.match_detected_to_annotated_peaks(detected, annotated, tolerance_samples=5)

    assert result["matched_annotated_idx"].tolist() == [0, 1]
    assert result["matched_detected_idx"].tolist() == [0, 2]
    assert result["unmatched_annotated_idx"].tolist() == []
    assert result["unmatched_detected_idx"].tolist() == [1]


def test_match_detected_to_annotated_peaks_respects_tolerance_boundary():
    """
    Distance exactly at the tolerance boundary matches; one sample past it does not.
    """
    within = cascade.match_detected_to_annotated_peaks(
        np.array([100]), np.array([105]), tolerance_samples=5
    )
    assert within["matched_annotated_idx"].tolist() == [0], "Distance == tolerance should match"

    outside = cascade.match_detected_to_annotated_peaks(
        np.array([100]), np.array([106]), tolerance_samples=5
    )
    assert outside["matched_annotated_idx"].tolist() == [], "Distance > tolerance should not match"
    assert outside["unmatched_annotated_idx"].tolist() == [0]
    assert outside["unmatched_detected_idx"].tolist() == [0]


def test_match_detected_to_annotated_peaks_reports_unmatched_on_both_sides():
    result = cascade.match_detected_to_annotated_peaks(
        np.array([50, 150]), np.array([1000]), tolerance_samples=5
    )
    assert result["unmatched_annotated_idx"].tolist() == [0]
    assert result["unmatched_detected_idx"].tolist() == [0, 1]


class _FakeBinaryModel:
    def __init__(self, probs):
        self._probs = np.asarray(probs, dtype=np.float32).reshape(-1, 1)

    def predict(self, x, batch_size=None, verbose=0):
        assert len(x) == len(self._probs), "Model called with unexpected number of windows"
        return self._probs


class _FakeMulticlassModel:
    def __init__(self):
        self.calls = []

    def predict(self, x, batch_size=None, verbose=0):
        self.calls.append(len(x))
        probs = np.zeros((len(x), 3), dtype=np.float32)
        probs[:, 0] = 1.0
        return probs


def test_run_cascade_batch_gates_stage_two_on_binary_threshold():
    """
    Only windows the binary model flags as anomalous (>= threshold) should ever
    reach the multiclass model - this is the batching optimization run_stream
    was refactored around, so it must not regress into scoring every beat twice.
    """
    fake_peaks = np.array([200, 400, 600, 800])

    signal = np.sin(np.linspace(0, 20, 1000)).astype(np.float32)
    scaler = StandardScaler().fit(np.random.default_rng(0).normal(size=(50, 216)))
    binary_model = _FakeBinaryModel(probs=[0.9, 0.1, 0.8, 0.2])
    multiclass_model = _FakeMulticlassModel()

    result = cascade.run_cascade_batch(
        signal, fs=250.0, binary_model=binary_model, scaler=scaler,
        multiclass_model=multiclass_model, window_size=216, binary_threshold=0.5,
        peak_detector=lambda signal, fs: fake_peaks,
    )

    assert result["total_beats"] == 4
    assert multiclass_model.calls == [2], "Expected exactly the 2 anomalous windows to reach stage 2"

    anomaly_flags = [beat["is_anomaly"] for beat in result["results"]]
    assert anomaly_flags == [True, False, True, False]

    predicted_classes = [beat["predicted_class"] for beat in result["results"]]
    assert predicted_classes[1] is None and predicted_classes[3] is None
    assert predicted_classes[0] is not None and predicted_classes[2] is not None


def test_run_cascade_batch_rejects_non_finite_signal():
    signal = np.array([1.0, np.nan, 3.0], dtype=np.float32)
    with pytest.raises(ValueError):
        cascade.run_cascade_batch(signal, fs=250.0, binary_model=None, scaler=None)

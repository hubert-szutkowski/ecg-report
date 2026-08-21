import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from pan_tompkins import neurokit_detect, pan_tompkins_detect

neurokit2 = pytest.importorskip("neurokit2", reason="neurokit_detect requires the neurokit2 package")


def _synthetic_ecg_like_signal(fs=360, n_beats=20, beat_period_sec=0.8):
    """A crude but peaky synthetic signal - enough to sanity-check that neurokit_detect returns
    plausible, monotonically increasing sample indices roughly spaced like real beats, without
    needing real MIT-BIH data (not available in this environment - see notebooks/loader.ipynb
    for the local path used for the real comparison)."""
    rng = np.random.default_rng(0)
    n_samples = int(n_beats * beat_period_sec * fs) + fs
    signal = rng.normal(0, 0.02, n_samples).astype(np.float32)
    beat_period_samples = int(beat_period_sec * fs)
    for i in range(n_beats):
        center = fs // 2 + i * beat_period_samples
        width = 6
        pulse = np.exp(-0.5 * ((np.arange(-width, width) / (width / 3)) ** 2))
        start = center - width
        end = start + len(pulse)
        if 0 <= start and end <= n_samples:
            signal[start:end] += pulse
    return signal, fs, beat_period_samples


def test_neurokit_detect_returns_sorted_sample_indices():
    signal, fs, _ = _synthetic_ecg_like_signal()
    peaks = neurokit_detect(signal, fs)

    assert isinstance(peaks, np.ndarray)
    assert peaks.ndim == 1
    assert np.all(np.diff(peaks) > 0)  # strictly increasing, no duplicates
    assert np.all((peaks >= 0) & (peaks < len(signal)))


def test_neurokit_detect_finds_roughly_the_expected_number_of_beats():
    signal, fs, beat_period_samples = _synthetic_ecg_like_signal(n_beats=20, beat_period_sec=0.8)
    peaks = neurokit_detect(signal, fs)

    # Loose bound - this is a synthetic signal, not real ECG, just checking it's in the right
    # ballpark rather than returning something degenerate (0 peaks, or thousands of them).
    assert 10 <= len(peaks) <= 30


def test_neurokit_detect_same_interface_as_pan_tompkins_detect():
    signal, fs, _ = _synthetic_ecg_like_signal()

    nk_peaks = neurokit_detect(signal, fs)
    pt_peaks = pan_tompkins_detect(signal, fs)

    assert nk_peaks.dtype.kind in ("i", "u")
    assert pt_peaks.dtype.kind in ("i", "u")
    assert nk_peaks.ndim == pt_peaks.ndim == 1

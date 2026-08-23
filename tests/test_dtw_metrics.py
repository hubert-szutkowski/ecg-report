import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/WGAN-GP')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/MultiClass')))

# gan_train pulls in tensorflow/mlflow/tslearn/dtaidistance at import time - only the `dl` env has
# them, so this module skips cleanly in the lightweight CI/test environment.
gan_train = pytest.importorskip(
    "gan_train", reason="dtw metric tests need the full GAN dependency set (tensorflow, dtaidistance, ...)"
)


def _signals(n, seed=0):
    """n beat-shaped windows, shaped like the (n, window, 1) arrays the real callers pass."""
    rng = np.random.default_rng(seed)
    base = np.sin(np.linspace(0, 4 * np.pi, 216))
    return (base[None, :] * rng.uniform(0.5, 1.5, size=(n, 1)) + rng.normal(0, 0.05, size=(n, 216)))[..., None]


def test_dtw_within_set_is_deterministic_when_subsampling():
    """
    Regression guard: dtw_within_set used np.random.choice with no seed, so any set larger than
    max_samples produced a different number on every call. The same unchanged real data measured
    5.87 and 7.67 across two runs of class S - noise big enough to swamp genuine differences in
    the diversity_ratio those numbers feed, including the metric a HyperDrive sweep selects on.
    """
    signals = _signals(200)  # > max_samples (100), so the subsampling path is exercised

    first = gan_train.dtw_within_set(signals)["mean"]
    second = gan_train.dtw_within_set(signals)["mean"]

    assert first == second, "repeated calls on identical data must return identical values"


def test_dtw_within_set_seed_actually_controls_the_subsample():
    """
    Guards against the determinism above coming from something other than the seed - e.g. the
    subsampling branch being skipped entirely, which would make the test above pass vacuously.
    """
    signals = _signals(200)

    default_seed = gan_train.dtw_within_set(signals, random_state=42)["mean"]
    other_seed = gan_train.dtw_within_set(signals, random_state=7)["mean"]

    assert default_seed != other_seed, "a different seed must select a different subsample"


def test_dtw_within_set_ignores_seed_below_the_subsampling_threshold():
    """Sets smaller than max_samples use every signal, so the seed cannot change the result."""
    signals = _signals(40)  # < max_samples

    assert gan_train.dtw_within_set(signals, random_state=42)["mean"] == gan_train.dtw_within_set(signals, random_state=7)["mean"]


def test_dtw_cross_set_is_deterministic_when_subsampling():
    real = _signals(200, seed=1)
    synthetic = _signals(200, seed=2)

    first = gan_train.dtw_cross_set(real, synthetic)["mean"]
    second = gan_train.dtw_cross_set(real, synthetic)["mean"]

    assert first == second

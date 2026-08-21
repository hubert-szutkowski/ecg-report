import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


def pan_tompkins_detect(signal: np.ndarray, fs: float, return_intermediate: bool = False):
    nyq = fs / 2.0
    low, high = 5 / nyq, 15 / nyq
    b, a = butter(1, [low, high], btype="band")
    filtered = filtfilt(b, a, signal)

    derivative = np.diff(filtered, prepend=filtered[0])
    squared = derivative ** 2

    integration_window = max(1, int(0.150 * fs))
    integrated = np.convolve(squared, np.ones(integration_window) / integration_window, mode="same")

    min_distance = max(1, int(0.200 * fs))
    threshold = np.mean(integrated) + 0.5 * np.std(integrated)
    candidate_peaks, _ = find_peaks(integrated, distance=min_distance, height=threshold)

    refine_window = max(1, int(0.075 * fs))
    refined_peaks = []
    for p in candidate_peaks:
        start = max(0, p - refine_window)
        end = min(len(signal), p + refine_window)
        local_max = start + int(np.argmax(signal[start:end]))
        refined_peaks.append(local_max)

    peaks = np.unique(np.array(refined_peaks, dtype=int))
    if not return_intermediate:
        return peaks
    return peaks, {"filtered": filtered, "integrated": integrated, "threshold": threshold}


def neurokit_detect(signal: np.ndarray, fs: float) -> np.ndarray:
    """
    R-peak detector backed by neurokit2's 'neurokit' method. Same interface as
    pan_tompkins_detect(signal, fs) -> np.ndarray of detected peak sample indices, so it's a
    drop-in alternative wherever a detector callable is expected (e.g. cascade.run_cascade_batch).

    Chosen after a multi-stage comparison (docs/peak_detection_benchmark_prompt.md,
    notebooks/outputs/peak_detector_cascade_comparison_report.md): clearly better isolated
    detector sensitivity than pan_tompkins_detect (0.92 vs 0.87 aggregate; nearly fixes record
    104, 0.55 -> 0.95), a patient-paired bootstrap on the full cascade puts P(better than
    pan_tompkins_detect) = 0.94, and detection latency is comparable (~70ms vs ~46ms/record).
    neurokit2's 'promac' ensemble was also tried and rejected: it doesn't fix record 108 (the
    other known-weak case - it's actually the worst of the three there) and its cascade-level
    advantage was less statistically supported than this method despite a higher point estimate.

    Not yet wired in as cascade.py's default detector - that requires a full retrain/
    re-evaluation cycle, a deliberate follow-up decision, not an automatic consequence of this
    function existing.

    Requires the neurokit2 package. Imported lazily here (not at module level) so importing
    pan_tompkins.py itself never requires it - this module stays dependency-light by default,
    matching src/cascade.py's design (see README's "Containerized Inference" section).
    """
    import neurokit2 as nk

    _, info = nk.ecg_peaks(signal, sampling_rate=fs, method="neurokit")
    return np.asarray(info["ECG_R_Peaks"])

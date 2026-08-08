import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


def pan_tompkins_detect(signal: np.ndarray, fs: float) -> np.ndarray:
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

    return np.unique(np.array(refined_peaks, dtype=int))

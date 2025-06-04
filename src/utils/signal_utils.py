# src/utils/signal_utils.py

import numpy as np
from scipy.signal import butter, lfilter, welch, find_peaks, coherence, hilbert, iirnotch, filtfilt
from scipy.interpolate import interp1d # Kept, though not immediately used below

# --- EEG Specific Filters & Processing ---
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """Applies a zero-phase Butterworth bandpass filter."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    if low <= 0 or high >= 1 or low >= high:
        print(f"Warning: Invalid critical frequencies for bandpass ({lowcut}-{highcut} Hz at {fs} Hz sampling). Low: {low}, High: {high}. Returning original data.")
        return data.copy() # Return a copy to avoid modifying original on error
    try:
        b, a = butter(order, [low, high], btype='band', analog=False)
        padlen = min(len(data) - 1, int(1.5 * max(len(b), len(a)))) # SciPy default is 3*max(len(a),len(b)), can be too large for short segments
        if len(data) <= padlen: # Ensure data is longer than padlen
             # print(f"Warning: Data length ({len(data)}) <= padlen ({padlen}) for filtfilt. Using lfilter (phase shift will occur).")
             return lfilter(b, a, data)
        y = filtfilt(b, a, data, padlen=padlen)
    except ValueError as e:
        # print(f"Error during bandpass filtfilt ({lowcut}-{highcut} Hz): {e}. Data len: {len(data)}. Using lfilter.")
        try:
            y = lfilter(b, a, data) # Fallback to lfilter, which will have phase shift
        except ValueError: # If lfilter also fails (e.g. data too short for filter order)
            # print(f"lfilter also failed. Returning original data for bandpass.")
            return data.copy()
    return y

def apply_notch_filter(data, notch_freq, quality_factor, fs):
    """Applies a zero-phase notch filter."""
    if notch_freq <= 0 or notch_freq >= fs / 2:
        return data.copy()
    try:
        b_notch, a_notch = iirnotch(w0=notch_freq, Q=quality_factor, fs=fs)
        padlen = min(len(data) - 1, int(1.5 * max(len(b_notch), len(a_notch))))
        if len(data) <= padlen:
            # print(f"Warning: Data length ({len(data)}) <= padlen ({padlen}) for notch filtfilt. Using lfilter (phase shift will occur).")
            return lfilter(b_notch, a_notch, data)
        y = filtfilt(b_notch, a_notch, data, padlen=padlen)
    except ValueError as e:
        # print(f"Error during notch filtfilt ({notch_freq} Hz): {e}. Data len: {len(data)}. Using lfilter.")
        try:
            y = lfilter(b_notch, a_notch, data)
        except ValueError:
            # print(f"lfilter also failed. Returning original data for notch.")
            return data.copy()
    return y

def calculate_psd_welch(data, fs, nperseg, noverlap=None, window='hann'):
    """Calculates PSD using Welch's method."""
    if noverlap is None: noverlap = nperseg // 2
    if len(data) < nperseg: nperseg = len(data); noverlap = 0
    if nperseg == 0: return np.array([]), np.array([])
    try:
        freqs, psd = welch(data, fs, window=window, nperseg=nperseg, noverlap=noverlap)
    except ValueError: return np.array([]), np.array([]) # Handles issues like empty data after windowing
    return freqs, psd

def calculate_band_power(freqs, psd, band, total_psd_power_in_ref_band=None):
    """Calculates absolute or relative power in a frequency band using trapezoidal integration."""
    if freqs.size == 0 or psd.size == 0: return 0.0
    idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
    if not np.any(idx_band): return 0.0
    
    # Ensure at least two points for trapz; otherwise, sum (less accurate but won't fail)
    if np.sum(idx_band) < 2:
        if np.sum(idx_band) == 1 and len(freqs)>1: # Approx if only one point in band
            df = np.mean(np.diff(freqs)) if len(freqs) > 1 else 1.0 # Approximate bin width
            power = psd[idx_band][0] * df
        elif np.sum(idx_band) == 1 and len(freqs)==1: # Only one freq point overall
             power = psd[idx_band][0]
        else: power = 0.0
    else:
        power = np.trapz(psd[idx_band], freqs[idx_band])

    if total_psd_power_in_ref_band is not None:
        return power / (total_psd_power_in_ref_band + 1e-12) # Add epsilon for safety
    return power

def calculate_spectral_edge_frequency(freqs, psd, total_power_in_ref_band, sef_percentage=0.95):
    """Calculates Spectral Edge Frequency (SEF)."""
    if freqs.size == 0 or psd.size == 0 or total_power_in_ref_band <= 1e-12: return np.nan
    
    # Ensure psd is non-negative for cumulative sum
    psd_non_negative = np.maximum(0, psd)
    
    # Calculate cumulative power using trapezoidal rule for each frequency bin
    cumulative_power_values = np.zeros_like(freqs)
    if len(freqs) > 1:
        for i in range(1, len(freqs)):
            cumulative_power_values[i] = np.trapz(psd_non_negative[:i+1], freqs[:i+1])
        if len(freqs) == 1 : # single point, use its power
             cumulative_power_values[0] = psd_non_negative[0] * (freqs[0] if freqs[0] > 0 else 1.0) # approx bin width if only one freq
    elif len(freqs) == 1:
        cumulative_power_values[0] = psd_non_negative[0]
    else: return np.nan


    sef_power_target = sef_percentage * total_power_in_ref_band
    
    edge_indices = np.where(cumulative_power_values >= sef_power_target)[0]
    if edge_indices.any():
        sef_idx = edge_indices[0]
        if sef_idx > 0 and cumulative_power_values[sef_idx] > cumulative_power_values[sef_idx - 1]:
            f_prev, f_curr = freqs[sef_idx-1], freqs[sef_idx]
            p_prev, p_curr = cumulative_power_values[sef_idx-1], cumulative_power_values[sef_idx]
            sef = f_prev + (sef_power_target - p_prev) * (f_curr - f_prev) / (p_curr - p_prev + 1e-12) # Epsilon for safety
            return np.clip(sef, freqs[0], freqs[-1])
        return freqs[sef_idx]
    return freqs[-1] if freqs.size > 0 else np.nan


def calculate_coherence_eeg(signal1, signal2, fs, band, nperseg_coh, noverlap_coh=None):
    """Calculates mean coherence in a band for EEG."""
    if noverlap_coh is None: noverlap_coh = nperseg_coh // 2
    if len(signal1) < nperseg_coh or len(signal2) < nperseg_coh: return np.nan
    try:
        freqs, Cxy = coherence(signal1, signal2, fs=fs, window='hann', nperseg=nperseg_coh, noverlap=noverlap_coh)
    except ValueError: return np.nan # Handles issues with segment lengths
    idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
    if not np.any(idx_band): return np.nan
    return np.mean(Cxy[idx_band])

def calculate_plv_eeg(signal1_bandpassed, signal2_bandpassed):
    """Calculates PLV. Assumes signals are already filtered to the band of interest."""
    if len(signal1_bandpassed) != len(signal2_bandpassed) or len(signal1_bandpassed) == 0: return np.nan
    phase1 = np.angle(hilbert(signal1_bandpassed))
    phase2 = np.angle(hilbert(signal2_bandpassed))
    phase_diff = phase1 - phase2
    return np.abs(np.mean(np.exp(1j * phase_diff)))

# --- PPG Specific Filters & Processing --- (Matches your uploaded signal_utils.py)
def butter_lowpass_filter_ppg(data, cutoff, fs, order=3):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    if normal_cutoff >= 1.0 or normal_cutoff <=0: return data.copy()
    try:
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        padlen = min(len(data)-1, int(1.5 * max(len(b),len(a))))
        if len(data) <= padlen: return lfilter(b, a, data)
        y = filtfilt(b, a, data, padlen=padlen)
    except ValueError: return lfilter(b, a, data) if len(data) > order else data.copy()
    return y

def find_ppg_peaks(cleaned_ppg_signal, fs, min_beat_interval_sec=0.35, max_beat_interval_sec=1.8):
    if len(cleaned_ppg_signal) < fs * min_beat_interval_sec: return np.array([])
    min_dist_samples = int(min_beat_interval_sec * fs)
    signal_median = np.median(cleaned_ppg_signal) # More robust than mean to outliers
    signal_std = np.std(cleaned_ppg_signal)
    min_height = signal_median # Simple threshold, can be improved
    # min_prominence = 0.1 * signal_std # Example: can be tuned

    peaks, _ = find_peaks(cleaned_ppg_signal, distance=min_dist_samples, height=min_height) # Removed prominence for simplicity now, can add back
    return peaks

def calculate_ibi_from_peaks(peak_indices, fs): # Renamed from calculate_ibi
    if len(peak_indices) < 2: return np.array([])
    return np.diff(peak_indices) * (1000.0 / fs)

def clean_ibi_series(ibi_ms, min_ibi_ms=300, max_ibi_ms=1800, ectopic_threshold_ratio=0.25): # Adjusted threshold
    if ibi_ms.size < 2: return ibi_ms # Need at least two to compare
    # Remove physiologically implausible IBIs
    cleaned_ibi = ibi_ms[(ibi_ms >= min_ibi_ms) & (ibi_ms <= max_ibi_ms)]
    if len(cleaned_ibi) < 3: return cleaned_ibi # Need at least 3 for median-based ectopic removal
    
    # Iterative ectopic removal can be better, but for a simple version:
    median_nn = np.median(cleaned_ibi)
    abs_diff_from_median = np.abs(cleaned_ibi - median_nn)
    # Keep IBIs that are within a certain percentage of the median
    cleaned_ibi = cleaned_ibi[abs_diff_from_median < ectopic_threshold_ratio * median_nn]
    return cleaned_ibi

def calculate_hrv_time_domain(cleaned_ibi_ms_segment): # Renamed from calculate_hrv_metrics
    """Calculates RMSSD, SDNN, MeanNN, pNN50 from a segment of cleaned IBIs."""
    n_ibi = len(cleaned_ibi_ms_segment)
    if n_ibi < 2: # Need at least 2 IBIs for diff (for RMSSD, pNN50), more for stable SDNN/MeanNN
        return {'HRV_RMSSD': np.nan, 'HRV_SDNN': np.nan, 'HRV_MeanNN': np.nan, 'HRV_pNN50': np.nan}

    nn_intervals = cleaned_ibi_ms_segment
    diff_nn = np.diff(nn_intervals)
    
    rmssd = np.sqrt(np.mean(diff_nn**2)) if len(diff_nn) > 0 else np.nan
    sdnn = np.std(nn_intervals) if n_ibi > 1 else np.nan # std needs at least 2 points for non-zero
    mean_nn = np.mean(nn_intervals) if n_ibi > 0 else np.nan
    pnn50 = np.sum(np.abs(diff_nn) > 50) / len(diff_nn) * 100 if len(diff_nn) > 0 else np.nan
    
    return {'HRV_RMSSD': rmssd, 'HRV_SDNN': sdnn, 'HRV_MeanNN': mean_nn, 'HRV_pNN50': pnn50}

# --- IMU Specific Filters & Processing --- (Matches your uploaded signal_utils.py)
def calculate_vector_magnitude(acc_x, acc_y, acc_z):
    return np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)

def calculate_imu_stillness(accel_mag_segment, gyro_mag_segment=None, epsilon=1e-9): # Renamed from calculate_imu_variance
    if accel_mag_segment is None or len(accel_mag_segment) < 2: return np.nan
    var_accel_mag = np.var(accel_mag_segment)
    # Higher score = more still (inverse of variance)
    stillness_score = 1.0 / (var_accel_mag + epsilon) 
    # You might want to cap this score if variance is extremely small, leading to huge stillness scores
    # stillness_score = np.clip(stillness_score, 0, 1000) # Example cap
    return stillness_score
# src/data_processing/ppg_processor.py
import numpy as np
from scipy.signal import butter, filtfilt, detrend
from src.utils.config_manager import load_config
from src.utils.signal_utils import (butter_lowpass_filter_ppg, find_ppg_peaks, calculate_ibi_from_peaks, clean_ibi_series)

class PPGProcessor:
    def __init__(self):
        config = load_config()
        self.ppg_settings = config['signal_processing']['ppg']
        
        # Basic parameters
        self.fs = self.ppg_settings['sample_rate']
        self.window_sec = self.ppg_settings['hrv_window_sec']
        self.overlap_ratio = self.ppg_settings['feature_overlap_ratio']
        self.window_samples = int(self.window_sec * self.fs)
        self.step_samples = int(self.window_samples * (1.0 - self.overlap_ratio))
        
        # Filter parameters
        self.lowpass_cutoff = self.ppg_settings.get('lowpass_cutoff_hz', 5.0) 
        self.filter_order = self.ppg_settings.get('ppg_filter_order', 3)
        
        # Heart rate/IBI parameters
        self.min_beat_interval_sec = self.ppg_settings.get('min_beat_interval_sec', 0.33)  # ~180 bpm max
        self.max_beat_interval_sec = self.ppg_settings.get('max_beat_interval_sec', 2.0)   # ~30 bpm min
        self.min_ibi_ms = self.min_beat_interval_sec * 1000
        self.max_ibi_ms = self.max_beat_interval_sec * 1000
        self.ibi_ectopic_threshold = self.ppg_settings.get('ibi_ectopic_threshold_ratio', 0.25)
        self.min_peak_distance = int(self.ppg_settings.get('min_peak_distance_sec', self.min_beat_interval_sec) * self.fs)
        
        print(f"PPGProcessor Initialized.")
        print(f"  PPG Fs: {self.fs} Hz, Lowpass Cutoff: {self.lowpass_cutoff} Hz")
        print(f"  Epoching: {self.window_sec}s window, {self.step_samples/self.fs:.2f}s step")
        print(f"  HR range: {60/self.max_beat_interval_sec:.0f}-{60/self.min_beat_interval_sec:.0f} BPM")
    
    def preprocess_ppg(self, raw_ppg_data):
        """Apply basic preprocessing to raw PPG data"""
        # Handle proper channel selection - GREEN is typically best for HRV
        if raw_ppg_data.ndim == 2:
            if raw_ppg_data.shape[0] == 3:  # (channels, samples)
                green_channel = raw_ppg_data[0, :]
            elif raw_ppg_data.shape[1] == 3:  # (samples, channels)
                green_channel = raw_ppg_data[:, 0]
            else:
                # Handle single-channel PPG with shape detection
                green_channel = raw_ppg_data.ravel()  # Flatten to 1D
        else:
            # Already 1D
            green_channel = raw_ppg_data
    
    def preprocess_and_extract_ibis(self, raw_ppg_channel_data):
        """
        Cleans a single channel of raw PPG data, detects peaks, calculates, and cleans IBIs.
        Args:
            raw_ppg_channel_data (np.ndarray): 1D array of raw PPG signal.
        Returns:
            tuple: (
                cleaned_filtered_ppg_signal (np.ndarray): The filtered PPG signal.
                peak_indices_session (np.ndarray): Sample indices of detected R-R peaks.
                cleaned_ibi_series_ms_session (np.ndarray): Cleaned Inter-Beat Intervals in milliseconds.
            )
        """
        if raw_ppg_channel_data is None or len(raw_ppg_channel_data) < self.fs * self.max_beat_interval_sec:
            return np.array([]), np.array([]), np.array([])

        # 1. Filtering 
        filtered_ppg = self.preprocess_ppg(raw_ppg_channel_data)
        
        # 2. Peak Detection
        peak_indices = find_ppg_peaks(filtered_ppg, self.fs, 
                                    min_beat_interval_sec=self.min_beat_interval_sec,
                                    max_beat_interval_sec=self.max_beat_interval_sec)
        
        if len(peak_indices) < 2:
            return filtered_ppg, peak_indices, np.array([])

        # 3. IBI Calculation from detected peaks
        ibi_ms_raw = calculate_ibi_from_peaks(peak_indices, self.fs)
        
        # 4. IBI Cleaning (remove ectopic beats, artifacts)
        cleaned_ibi_ms = clean_ibi_series(ibi_ms_raw, 
                                        min_ibi_ms=self.min_ibi_ms, 
                                        max_ibi_ms=self.max_ibi_ms,
                                        ectopic_threshold_ratio=self.ibi_ectopic_threshold)
        
        return filtered_ppg, peak_indices, cleaned_ibi_ms
    
    def generate_ppg_epochs(self, ppg_data_trimmed):
        """Generate epochs from trimmed PPG data"""
        if ppg_data_trimmed is None or len(ppg_data_trimmed) < self.window_samples:
            return
        
        total_samples = len(ppg_data_trimmed)
        failure_reasons = {"length": 0, "no_peaks": 0, "filtering": 0}
        
        for start_idx in range(0, total_samples - self.window_samples + 1, self.step_samples):
            current_window = ppg_data_trimmed[start_idx : start_idx + self.window_samples]
            yield current_window, start_idx / self.fs
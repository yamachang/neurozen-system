# src/data_processing/eeg_processor.py
import numpy as np
from scipy import signal
from scipy.signal import butter, iirnotch, lfilter, sosfilt, savgol_filter
from src.utils.signal_utils import butter_bandpass_filter, apply_notch_filter
from src.utils.config_manager import load_config

class EEGProcessor:
    """
    Processes EEG data for feature extraction using a fast causal preprocessing pipeline.
    ML-COMPATIBLE: Uses consistent preprocessing for both training and real-time inference.
    
    Key Features:
    - Signal-level normalization for high-amplitude channels (ML-compatible)
    - Fast causal filtering pipeline
    - Consistent processing between training and inference
    """
    def __init__(self):
        config = load_config()
        self.eeg_settings = config['signal_processing']['eeg']
        
        self.fs = self.eeg_settings['sample_rate']
        self.active_channel_indices = self.eeg_settings['active_channels_indices']
        self.active_channel_names = self.eeg_settings.get('active_channels_names', [f"Ch{i}" for i in self.active_channel_indices])
        self.num_active_channels = len(self.active_channel_indices)
        self.total_device_channels = self.eeg_settings['total_channels_from_device']

        # Ensure active_channel_names has correct length if not perfectly matching indices length
        if len(self.active_channel_names) != self.num_active_channels:
            self.active_channel_names = [f"ActiveCh{i+1}" for i in range(self.num_active_channels)]

        # Causal preprocessing parameters
        self.notch_freq = self.eeg_settings.get('notch_freq', 60.0)
        self.notch_quality_factor = self.eeg_settings.get('notch_quality_factor', 30.0)
        self.bandpass_lowcut = 1.0  # Fixed for new pipeline
        self.bandpass_highcut = 40.0  # Fixed for new pipeline
        self.filter_order = 4  # Fixed for new pipeline
        
        # Rolling statistics parameters
        self.rolling_window_sec = 30.0
        self.mad_threshold = 6.0
        self.median_filter_size = 5
        self.slope_threshold_factor = 6.0
        
        # Epoch quality assessment parameters (configurable)
        self.epoch_artifact_threshold = self.eeg_settings.get('epoch_artifact_threshold_pct', 15.0)  # Allow up to 15% artifacts
        self.enable_session_level_gating = self.eeg_settings.get('enable_session_level_epoch_gating', True)
        
        # Legacy parameters (kept for compatibility)
        self.artifact_thresh_uv = self.eeg_settings['artifact_amplitude_threshold_uv']
        self.uv_scaling_factor = self.eeg_settings.get('eeg_uv_scaling_factor', 1.0)

        self.window_sec = self.eeg_settings['feature_window_sec']
        self.overlap_ratio = self.eeg_settings['feature_overlap_ratio']
        self.window_samples = int(self.window_sec * self.fs)
        self.step_samples = int(self.window_samples * (1.0 - self.overlap_ratio))
        
        # Define channel-specific SQC thresholds
        self.sqc_good_threshold = self.eeg_settings.get('sqc_good_threshold', 1)
        self.channel_specific_sqc_thresholds = {
            'LF': self.sqc_good_threshold,
            'OTEL': self.sqc_good_threshold,
            # Set different thresholds for problem channels
            'RF': 0,     # Ignoring SQC for this channel
            'OTER': 0    # Ignoring SQC for this channel
        }
        
        # Identify high-amplitude channels for special processing
        self.high_amplitude_channels = ['RF', 'OTER']
        
        self.fs_eeg_to_sqc_ratio = self.eeg_settings.get('sqc_fs_divisor', self.fs)

        # Initialize rejection statistics tracking
        self.rejection_stats = {
            "total_epochs": 0,
            "standard_epochs": 0,
            "special_processed_epochs": 0,
            "external_sqc_failures": 0,
            "flatline_failures": 0,
            "amplitude_failures": 0,
            "channel_specific_failures": {ch: 0 for ch in self.active_channel_names},
            "sqc_channel_failures": {ch: 0 for ch in self.active_channel_names}
        }
        
        # Initialize causal filters (will be created per session)
        self.causal_filters = None
        self.rolling_stats_initialized = False
        self.rolling_window_size = int(self.rolling_window_sec * self.fs)
        self.update_interval = max(1, self.fs // 4)  # Update stats 4 times per second

        print(f"EEGProcessor Initialized with ML-Compatible Fast Causal Pipeline.")
        print(f"  Total device channels expected: {self.total_device_channels}")
        print(f"  Processing {self.num_active_channels} active channels: {self.active_channel_names} (Indices from raw: {self.active_channel_indices})")
        print(f"  Special processing for high-amplitude channels: {self.high_amplitude_channels}")
        print(f"  EEG Fs: {self.fs} Hz. Scaling factor to uV: {self.uv_scaling_factor}.")
        print(f"  Causal Pipeline: Notch {self.notch_freq} Hz, Bandpass {self.bandpass_lowcut}-{self.bandpass_highcut} Hz")
        print(f"  Rolling stats: {self.rolling_window_sec}s window, MAD threshold: {self.mad_threshold}, Median filter: {self.median_filter_size} samples")
        print(f"  Epoch quality: {self.epoch_artifact_threshold:.1f}% artifact threshold, Session gating: {'Enabled' if self.enable_session_level_gating else 'Disabled'}")
        print(f"  SQC Good Threshold: {self.sqc_good_threshold}. Channel-specific thresholds: {self.channel_specific_sqc_thresholds}")
        print(f"  EEG Fs to SQC Fs Divisor: {self.fs_eeg_to_sqc_ratio} (Effective SQC Fs: {self.fs/self.fs_eeg_to_sqc_ratio:.2f} Hz).")
        print(f"  Epoching: {self.window_sec}s window, {self.step_samples/self.fs:.2f}s step ({self.window_samples} samples epoch, {self.step_samples} samples step).")
        print(f"  ML-COMPATIBLE: Signal preprocessing identical for training and real-time inference")

    def _select_and_scale_eeg_data(self, raw_eeg_all_device_channels_window):
        if raw_eeg_all_device_channels_window.shape[0] != self.total_device_channels:
            if raw_eeg_all_device_channels_window.shape[0] == self.num_active_channels:
                selected_eeg = raw_eeg_all_device_channels_window 
            else: 
                return None 
        else:
            try: selected_eeg = raw_eeg_all_device_channels_window[self.active_channel_indices, :]
            except IndexError: 
                return None
        return selected_eeg * self.uv_scaling_factor

    def _apply_dc_baseline_correction(self, signal_data, window_length=501):
        """Apply moving baseline correction to handle DC offset with dynamic window size"""
        # Ensure window length is appropriate for signal length
        signal_length = len(signal_data)
        
        # Adjust window length if needed
        if window_length >= signal_length:
            # If signal is too short, use a smaller window
            # Make it odd, as required by savgol_filter (at most 1/4 of signal length)
            window_length = min(signal_length - 1, 101)
            if window_length % 2 == 0:
                window_length -= 1
            
            # Handle case where signal is extremely short
            if window_length < 3:
                # For very short signals, just subtract the mean
                return signal_data - np.mean(signal_data)
        
        try:
            # Use Savitzky-Golay filter to estimate the baseline drift
            baseline = savgol_filter(signal_data, window_length, 2)
            # Return the signal with baseline removed
            return signal_data - baseline
        except:
            # Fallback: simple detrend
            return signal_data - np.mean(signal_data)

    def _fast_rolling_median_mad(self, data, window_size, update_every=None):
        """
        Fast rolling median/MAD calculation with sparse updates
        Only recalculates every 'update_every' samples for efficiency
        """
        if update_every is None:
            update_every = self.update_interval
            
        n_samples = len(data)
        medians = np.zeros(n_samples)
        mads = np.ones(n_samples)  # Initialize to 1
        
        # Initialize first window
        if n_samples >= window_size:
            first_window = data[:window_size]
            medians[:window_size] = np.median(first_window)
            mads[:window_size] = np.median(np.abs(first_window - medians[0]))
            if mads[0] == 0:
                mads[:window_size] = 1.0
        
        # Update at sparse intervals
        for i in range(window_size, n_samples, update_every):
            end_idx = min(i + update_every, n_samples)
            start_window = max(0, i - window_size)
            
            window_data = data[start_window:i]
            median_val = np.median(window_data)
            mad_val = np.median(np.abs(window_data - median_val))
            if mad_val == 0:
                mad_val = 1.0
            
            # Fill the interval
            medians[i:end_idx] = median_val
            mads[i:end_idx] = mad_val
        
        return medians, mads

    def _fast_causal_eeg_pipeline(self, data):
        """
        ML-COMPATIBLE fast vectorized causal EEG processing pipeline.
        
        Key ML Features:
        - Signal-level normalization for high-amplitude channels (consistent across training/inference)
        - Causal filtering (works for real-time)
        - Deterministic processing (same input -> same output)
        
        Args:
            data: EEG data (channels x samples)
            
        Returns:
            processed_data: Cleaned EEG data
            artifact_flags: Boolean array indicating artifacts
            stats: Dictionary with processing statistics
        """
        num_channels, num_samples = data.shape
        processed_data = np.zeros_like(data)
        artifact_flags = np.zeros(data.shape, dtype=bool)
        
        # Design filters (create fresh for each session to avoid state issues)
        Q = self.notch_quality_factor
        notch_b, notch_a = iirnotch(self.notch_freq, Q, self.fs)
        bandpass_sos = butter(self.filter_order, [self.bandpass_lowcut, self.bandpass_highcut], 
                             btype='band', fs=self.fs, output='sos')
        
        channel_stats = {}
        
        for ch_idx in range(num_channels):
            ch_name = self.active_channel_names[ch_idx]
            channel_data = data[ch_idx, :].copy()
            
            # ===== ML-COMPATIBLE SIGNAL-LEVEL NORMALIZATION =====
            # Apply channel-specific preprocessing for high-amplitude channels
            # This is IDENTICAL for training and real-time inference
            if ch_name in self.high_amplitude_channels:
                print(f"    Applying ML-compatible signal normalization to {ch_name}")
                
                # Step 1: DC offset and drift correction
                channel_data = self._apply_dc_baseline_correction(channel_data)
                
                # Step 2: Z-score normalization (ML-compatible)
                mean_val = np.mean(channel_data)
                std_val = np.std(channel_data)
                if std_val > 1e-8:  # Avoid division by zero
                    channel_data = (channel_data - mean_val) / std_val
                
                # Step 3: Scale to reasonable amplitude range (similar to other channels)
                channel_data = channel_data * 10.0  # Scale to ±10 μV typical range
            
            # ===== CAUSAL FILTERING (ML-COMPATIBLE) =====
            # Step 1: Apply causal filters
            notch_filtered = lfilter(notch_b, notch_a, channel_data)
            bandpass_filtered = sosfilt(bandpass_sos, notch_filtered)
            
            # Step 2: Fast rolling median/MAD calculation
            medians, mads = self._fast_rolling_median_mad(bandpass_filtered, self.rolling_window_size, self.update_interval)
            
            # Step 3: Artifact detection and clipping
            lower_bounds = medians - self.mad_threshold * mads
            upper_bounds = medians + self.mad_threshold * mads
            
            # Find artifacts
            artifacts = (bandpass_filtered < lower_bounds) | (bandpass_filtered > upper_bounds)
            artifact_flags[ch_idx, :] = artifacts
            
            # Clip artifacts
            clipped_data = np.clip(bandpass_filtered, lower_bounds, upper_bounds)
            
            # Step 4: Simple causal median filter (running median of last N samples)
            if self.median_filter_size > 1:
                median_filtered = np.copy(clipped_data)
                for i in range(self.median_filter_size, num_samples):
                    window = clipped_data[i-self.median_filter_size+1:i+1]
                    median_filtered[i] = np.median(window)
                processed_channel = median_filtered
            else:
                processed_channel = clipped_data
            
            # Step 5: Slope check and correction
            if num_samples > 1:
                # Calculate differences
                diffs = np.diff(processed_channel)
                if len(diffs) > 0:
                    # Calculate rolling median and MAD of differences
                    diff_medians, diff_mads = self._fast_rolling_median_mad(diffs, 
                                                                          min(len(diffs), self.rolling_window_size), 
                                                                          self.update_interval)
                    
                    # Find slope outliers
                    slope_threshold = diff_medians + self.slope_threshold_factor * diff_mads
                    slope_outliers = np.abs(diffs) > slope_threshold
                    
                    # Correct slope outliers by reverting to previous value
                    corrected_data = np.copy(processed_channel)
                    for i in range(1, len(corrected_data)):
                        if slope_outliers[i-1]:  # diff[i-1] corresponds to change from i-1 to i
                            corrected_data[i] = corrected_data[i-1]
                    
                    processed_data[ch_idx, :] = corrected_data
                else:
                    processed_data[ch_idx, :] = processed_channel
            else:
                processed_data[ch_idx, :] = processed_channel
            
            # Store stats
            artifact_rate = np.mean(artifacts) * 100
            channel_stats[ch_idx] = {
                'artifact_rate': artifact_rate,
                'mean_mad': np.mean(mads),
                'mean_median': np.mean(medians),
                'signal_normalized': ch_name in self.high_amplitude_channels
            }
        
        # Calculate epoch quality (4-second epochs) with configurable threshold
        epoch_size = int(4 * self.fs)
        num_epochs = num_samples // epoch_size
        epoch_quality = []
        
        for epoch_idx in range(num_epochs):
            start_idx = epoch_idx * epoch_size
            end_idx = start_idx + epoch_size
            
            epoch_artifacts = artifact_flags[:, start_idx:end_idx]
            epoch_artifact_rate = np.mean(epoch_artifacts) * 100  # Convert to percentage
            
            # Use configurable threshold instead of fixed 5%
            epoch_is_good = epoch_artifact_rate < self.epoch_artifact_threshold
            epoch_quality.append(epoch_is_good)
        
        # Calculate statistics
        good_epochs_count = sum(epoch_quality)
        total_epochs_count = len(epoch_quality)
        good_epoch_pct = (good_epochs_count / max(1, total_epochs_count)) * 100
        
        stats = {
            'channel_stats': channel_stats,
            'epoch_quality': epoch_quality,
            'total_artifact_rate': np.mean(artifact_flags) * 100,
            'good_epochs': good_epochs_count,
            'total_epochs': total_epochs_count,
            'good_epoch_percentage': good_epoch_pct,
            'artifact_threshold_used': self.epoch_artifact_threshold,
            'signal_level_normalization_applied': any(ch in self.high_amplitude_channels for ch in self.active_channel_names)
        }
        
        return processed_data, artifact_flags, stats

    def _check_sqc_for_epoch(self, epoch_start_sample_in_trimmed_eeg, 
                             sqc_scores_for_active_channels_trimmed):
        if sqc_scores_for_active_channels_trimmed is None or sqc_scores_for_active_channels_trimmed.size == 0:
            return True, "No external SQC provided", []

        if sqc_scores_for_active_channels_trimmed.shape[0] != self.num_active_channels:
            return True, f"SQC channel count mismatch ({sqc_scores_for_active_channels_trimmed.shape[0]} vs {self.num_active_channels})", []

        start_sqc_idx = int(epoch_start_sample_in_trimmed_eeg / self.fs_eeg_to_sqc_ratio)
        end_sqc_idx_exclusive = int((epoch_start_sample_in_trimmed_eeg + self.window_samples) / self.fs_eeg_to_sqc_ratio)
        start_sqc_idx = max(0, start_sqc_idx)
        end_sqc_idx = max(start_sqc_idx, min(sqc_scores_for_active_channels_trimmed.shape[1], end_sqc_idx_exclusive))
        
        if start_sqc_idx >= end_sqc_idx:
             return True, "No SQC samples in epoch window", []

        epoch_sqc_window = sqc_scores_for_active_channels_trimmed[:, start_sqc_idx:end_sqc_idx]
        
        if epoch_sqc_window.size > 0:
            # Track which channels failed SQC, using channel-specific thresholds
            failed_channels = []
            critical_failed_channels = []
            
            for ch_idx_sqc in range(epoch_sqc_window.shape[0]):
                ch_name = self.active_channel_names[ch_idx_sqc]
                ch_threshold = self.channel_specific_sqc_thresholds.get(ch_name, self.sqc_good_threshold)
                
                if np.any(epoch_sqc_window[ch_idx_sqc,:] != ch_threshold):
                    failed_channels.append(ch_name)
                    
                    # Only consider non-high-amplitude channels as critical failures
                    if ch_name not in self.high_amplitude_channels:
                        critical_failed_channels.append(ch_name)
            
            # If any critical channels failed SQC, mark as failed
            if critical_failed_channels:
                return False, "External SQC failed in critical channels", failed_channels
            # If only high-amplitude channels failed, consider it a special case
            elif failed_channels:
                return True, "Only high-amplitude channels failed SQC", failed_channels
                
        return True, "External SQC passed", []

    def _preprocess_single_epoch_active_scaled(self, active_scaled_epoch_uv, 
                                             epoch_start_sample_in_trimmed_eeg,
                                             sqc_scores_for_active_channels_trimmed=None,
                                             session_processed_data=None,
                                             session_artifact_flags=None,
                                             session_stats=None):
        
        # Count this epoch in our statistics
        self.rejection_stats["total_epochs"] += 1
        
        external_sqc_passed, sqc_reason, failed_sqc_channels = self._check_sqc_for_epoch(
            epoch_start_sample_in_trimmed_eeg, 
            sqc_scores_for_active_channels_trimmed
        )
        
        # Track which channels failed SQC
        for ch in failed_sqc_channels:
            self.rejection_stats["sqc_channel_failures"][ch] += 1
        
        # If we have preprocessed session data, use it directly
        if session_processed_data is not None:
            start_idx = epoch_start_sample_in_trimmed_eeg
            end_idx = start_idx + self.window_samples
            
            if end_idx <= session_processed_data.shape[1]:
                processed_data = session_processed_data[:, start_idx:end_idx]
                
                # Check epoch quality from session stats (if enabled)
                special_processed = False
                epoch_quality = True  # Default to accepting epoch
                
                if (self.enable_session_level_gating and 
                    session_stats and 'epoch_quality' in session_stats):
                    epoch_idx = start_idx // int(4 * self.fs)  # 4-second epochs
                    if epoch_idx < len(session_stats['epoch_quality']):
                        epoch_quality = session_stats['epoch_quality'][epoch_idx]
                    
                    # If session-level gating rejects the epoch, provide detailed reason
                    if not epoch_quality:
                        artifact_pct = session_stats.get('artifact_threshold_used', 'unknown')
                        self.rejection_stats["amplitude_failures"] += 1
                        processed_data.fill(np.nan)
                        return processed_data, False, f"Session-level gating: >{artifact_pct}% artifacts", special_processed
                
                # Check for high-amplitude channel involvement
                if any(ch in failed_sqc_channels for ch in self.high_amplitude_channels):
                    special_processed = True
                
                if not external_sqc_passed:
                    self.rejection_stats["external_sqc_failures"] += 1
                    processed_data.fill(np.nan)
                    return processed_data, False, sqc_reason, special_processed
                
                # Update statistics based on processing type and quality
                if epoch_quality:
                    if special_processed:
                        self.rejection_stats["special_processed_epochs"] += 1
                    else:
                        self.rejection_stats["standard_epochs"] += 1
                    return processed_data, True, "ML-compatible causal pipeline processed", special_processed
                # If we reach here, epoch was rejected by session-level gating (handled above)
            else:
                # Epoch extends beyond processed data
                processed_data = np.full((self.num_active_channels, self.window_samples), np.nan)
                return processed_data, False, "Epoch extends beyond processed data", False
        
        # Fallback to legacy processing if no session data available
        # This should rarely happen with the new pipeline
        processed_data = np.zeros_like(active_scaled_epoch_uv)
        special_processed = False
        
        if not external_sqc_passed:
            self.rejection_stats["external_sqc_failures"] += 1
            processed_data.fill(np.nan)
            return processed_data, False, sqc_reason, special_processed
        
        # Apply basic quality checks
        internal_qc_passed = True
        internal_failure_reason = "Internal QC Passed"
        
        for i in range(self.num_active_channels):
            ch_name = self.active_channel_names[i] 
            channel_data = active_scaled_epoch_uv[i, :]
            
            # Skip QC for high-amplitude channels that only failed SQC
            skip_qc = ch_name in self.high_amplitude_channels and ch_name in failed_sqc_channels
            
            if not skip_qc:
                # Basic quality checks
                mean_abs_val = np.mean(np.abs(channel_data))
                std_val = np.std(channel_data)
                if mean_abs_val < 1e-4 or std_val < 1e-3: 
                    internal_qc_passed = False
                    internal_failure_reason = f"Flatline Ch {ch_name} (mean_abs={mean_abs_val:.2e}, std={std_val:.2e})"
                    self.rejection_stats["flatline_failures"] += 1
                    self.rejection_stats["channel_specific_failures"][ch_name] += 1
                    break 
                
                max_abs_amp_epoch_ch = np.max(np.abs(channel_data))
                if max_abs_amp_epoch_ch > self.artifact_thresh_uv:
                    internal_qc_passed = False
                    internal_failure_reason = f"Amplitude Ch {ch_name} ({max_abs_amp_epoch_ch:.0f}uV > {self.artifact_thresh_uv}uV)"
                    self.rejection_stats["amplitude_failures"] += 1
                    self.rejection_stats["channel_specific_failures"][ch_name] += 1
                    break
            
            # Use raw data for fallback processing
            processed_data[i, :] = channel_data
            
            # If we're processing a high-amplitude channel with special handling
            if ch_name in self.high_amplitude_channels and ch_name in failed_sqc_channels:
                special_processed = True
        
        if not internal_qc_passed:
            processed_data.fill(np.nan)
            return processed_data, False, internal_failure_reason, special_processed
        
        # Update statistics based on processing type
        if special_processed:
            self.rejection_stats["special_processed_epochs"] += 1
        else:
            self.rejection_stats["standard_epochs"] += 1
            
        return processed_data, True, "Fallback processing", special_processed

    def generate_preprocessed_epochs(self, session_trimmed_all_device_eeg_data, 
                                     sqc_scores_for_active_channels_trimmed=None):
        if session_trimmed_all_device_eeg_data is None or \
           session_trimmed_all_device_eeg_data.ndim != 2 or \
           session_trimmed_all_device_eeg_data.shape[1] < self.window_samples:
            return 

        self._epoch_fail_reasons_summary = {}
        total_samples_trimmed = session_trimmed_all_device_eeg_data.shape[1]

        # Apply ML-compatible session-level preprocessing using the fast causal pipeline
        print("  Applying ML-compatible fast causal EEG preprocessing to full session...")
        
        # Select and scale the active channels for session processing
        scaled_active_session_data = self._select_and_scale_eeg_data(session_trimmed_all_device_eeg_data)
        
        if scaled_active_session_data is None:
            print("  Error: Could not select/scale EEG data for session processing")
            return
        
        # Apply the fast causal pipeline to the entire session
        session_processed_data, session_artifact_flags, session_stats = self._fast_causal_eeg_pipeline(scaled_active_session_data)
        
        print(f"  Session preprocessing complete. Overall artifact rate: {session_stats['total_artifact_rate']:.2f}%")
        print(f"  Good epochs: {session_stats['good_epochs']}/{session_stats['total_epochs']} ({session_stats.get('good_epoch_percentage', 0):.1f}%) using {session_stats.get('artifact_threshold_used', 'default')}% threshold")
        if session_stats.get('signal_level_normalization_applied', False):
            print(f"  Signal-level normalization applied to high-amplitude channels")
        if not self.enable_session_level_gating:
            print(f"  Session-level epoch gating disabled - accepting all epochs that pass SQC")

        # Now generate epochs from the preprocessed session data
        for start_idx in range(0, total_samples_trimmed - self.window_samples + 1, self.step_samples):
            current_window_all_device_channels = session_trimmed_all_device_eeg_data[:, start_idx : start_idx + self.window_samples]
            scaled_active_epoch_uv = self._select_and_scale_eeg_data(current_window_all_device_channels)
            
            if scaled_active_epoch_uv is None:
                yield None, False, start_idx / self.fs, False
                self._epoch_fail_reasons_summary["Select/Scale Error"] = self._epoch_fail_reasons_summary.get("Select/Scale Error", 0) + 1
                continue

            processed_epoch, quality_flag, reason, special_processed = self._preprocess_single_epoch_active_scaled(
                scaled_active_epoch_uv,
                start_idx, 
                sqc_scores_for_active_channels_trimmed,
                session_processed_data,
                session_artifact_flags,
                session_stats
            )
            if not quality_flag:
                 self._epoch_fail_reasons_summary[reason] = self._epoch_fail_reasons_summary.get(reason, 0) + 1
            yield processed_epoch, quality_flag, start_idx / self.fs, special_processed
            
    # Add this method to your existing EEGProcessor class in src/data_processing/eeg_processor.py

    def preprocess_single_epoch_realtime(self, raw_epoch_data):
        """
        Preprocess a single EEG epoch for real-time processing.
        
        This method applies the same ML-compatible processing as the batch pipeline
        but works on individual epochs without requiring session-level preprocessing.
        
        Args:
            raw_epoch_data: Raw EEG epoch data (total_device_channels x samples) from all device channels
            
        Returns:
            tuple: (processed_epoch, quality_flag)
                - processed_epoch: Preprocessed EEG data (active_channels x samples)
                - quality_flag: Boolean indicating if epoch passes quality checks
        """
        try:
            # Debug: Show input data shape
            print(f"    Debug: Raw epoch shape: {raw_epoch_data.shape}")
            
            # Step 1: Select and scale active channels (this handles the 6->4 channel selection)
            scaled_active_epoch = self._select_and_scale_eeg_data(raw_epoch_data)
            
            if scaled_active_epoch is None:
                print(f"    Debug: Channel selection failed")
                return np.full((self.num_active_channels, self.window_samples), np.nan), False
            
            print(f"    Debug: Selected active channels shape: {scaled_active_epoch.shape}")
            print(f"    Debug: Processing channels: {self.active_channel_names}")
            
            # Step 2: Basic quality checks
            for ch_idx in range(scaled_active_epoch.shape[0]):
                ch_name = self.active_channel_names[ch_idx]
                channel_data = scaled_active_epoch[ch_idx, :]
                
                # Skip quality checks for high-amplitude channels (they get special processing)
                if ch_name not in self.high_amplitude_channels:
                    # Check for flatline
                    mean_abs_val = np.mean(np.abs(channel_data))
                    std_val = np.std(channel_data)
                    if mean_abs_val < 1e-4 or std_val < 1e-3:
                        print(f"    Debug: {ch_name} failed flatline check (mean_abs={mean_abs_val:.2e}, std={std_val:.2e})")
                        return np.full((self.num_active_channels, self.window_samples), np.nan), False
                    
                    # Check for extreme amplitudes
                    max_abs_amp = np.max(np.abs(channel_data))
                    if max_abs_amp > self.artifact_thresh_uv:
                        print(f"    Debug: {ch_name} failed amplitude check ({max_abs_amp:.0f}uV > {self.artifact_thresh_uv}uV)")
                        return np.full((self.num_active_channels, self.window_samples), np.nan), False
                else:
                    print(f"    Debug: {ch_name} will receive special signal-level normalization")
            
            # Step 3: Apply ML-compatible fast causal preprocessing
            processed_data, artifact_flags, stats = self._fast_causal_eeg_pipeline(scaled_active_epoch)
            
            # Step 4: Assess epoch quality
            # Use the same artifact threshold as session processing
            total_artifact_rate = np.mean(artifact_flags) * 100
            epoch_quality = total_artifact_rate < self.epoch_artifact_threshold
            
            print(f"    Debug: Artifact rate: {total_artifact_rate:.1f}% (threshold: {self.epoch_artifact_threshold:.1f}%)")
            print(f"    Debug: Epoch quality: {'PASS' if epoch_quality else 'FAIL'}")
            
            return processed_data, epoch_quality
            
        except Exception as e:
            print(f"Error in real-time epoch preprocessing: {e}")
            import traceback
            traceback.print_exc()
            return np.full((self.num_active_channels, self.window_samples), np.nan), False

    def preprocess_epoch_simple(self, eeg_epoch):
        """
        Simplified preprocessing method for backwards compatibility.
        
        Args:
            eeg_epoch: EEG epoch data (channels x samples)
            
        Returns:
            Preprocessed EEG epoch data
        """
        processed_epoch, quality_flag = self.preprocess_single_epoch_realtime(eeg_epoch)
        return processed_epoch if quality_flag else None
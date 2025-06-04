# src/feature_extraction/ppg_features.py
import numpy as np
from scipy import signal, interpolate
from src.utils.config_manager import load_config

class PPGFeatureExtractor:
    def __init__(self):
        config = load_config()
        self.ppg_settings = config['signal_processing']['ppg']
        
        self.fs = self.ppg_settings['sample_rate']
        self.min_beat_interval_sec = self.ppg_settings.get('min_beat_interval_sec', 0.33)  # ~180 bpm
        self.max_beat_interval_sec = self.ppg_settings.get('max_beat_interval_sec', 2.0)   # ~30 bpm
        self.min_peak_distance = int(self.min_beat_interval_sec * self.fs)
        
        # Define PPG/HRV feature names
        self.feature_names = [
            'heart_rate',
            'sdnn',
            'rmssd',
            'pnn50',
            'lf_power', 
            'hf_power',
            'lf_hf_ratio'
        ]
        
        # Debug option - set to True to see more diagnostic information
        self.debug = False
    
    def extract_features_from_epoch(self, ppg_epoch):
        """Extract HRV features from a single PPG epoch with improved tolerance for shorter epochs"""
        features = {fname: np.nan for fname in self.feature_names}
        
        if ppg_epoch is None or len(ppg_epoch) == 0 or np.all(np.isnan(ppg_epoch)):
            if self.debug:
                print("  PPG Feature Extraction: Empty or NaN data")
            return features
        
        # Normalize and flatten if needed
        if hasattr(ppg_epoch, 'ndim') and ppg_epoch.ndim > 1:
            if self.debug:
                print(f"  PPG Feature Extraction: Flattening multidimensional input with shape {ppg_epoch.shape}")
            if ppg_epoch.shape[0] == 1:  # Single channel as row
                ppg_signal = ppg_epoch[0, :]
            elif ppg_epoch.shape[0] > 1 and ppg_epoch.shape[1] == 1:  # Single channel as column
                ppg_signal = ppg_epoch[:, 0]
            elif ppg_epoch.shape[0] == 3:  # Likely channels x samples (use green channel)
                ppg_signal = ppg_epoch[0, :]
            elif ppg_epoch.shape[1] == 3:  # Likely samples x channels (use green channel)
                ppg_signal = ppg_epoch[:, 0]
            else:
                ppg_signal = ppg_epoch.flatten()  # Just flatten if shape is unclear
        else:
            ppg_signal = ppg_epoch
        
        # Standardize signal to help with peak detection
        try:
            if np.std(ppg_signal) > 0:
                ppg_signal = (ppg_signal - np.mean(ppg_signal)) / np.std(ppg_signal)
        except:
            # Just continue if standardization fails
            pass
            
        try:
            # Try multiple peak detection approaches with increasing leniency
            # 1. First attempt: Standard peak finding with physiological constraints
            peaks, _ = signal.find_peaks(ppg_signal, distance=self.min_peak_distance)
            
            if len(peaks) < 2:  # Too few peaks with standard settings
                # 2. Second attempt: More lenient settings
                peaks, _ = signal.find_peaks(ppg_signal, distance=max(3, int(self.min_peak_distance * 0.8)))
                if self.debug and len(peaks) >= 2:
                    print(f"  PPG Feature Extraction: Using lenient peak detection, found {len(peaks)} peaks")
            
            if len(peaks) < 2:
                # 3. Final attempt: Use autocorrelation to estimate periodicity
                result = self.estimate_heart_rate_via_autocorrelation(ppg_signal)
                if result is not None:
                    features['heart_rate'] = result
                    if self.debug:
                        print(f"  PPG Feature Extraction: Used autocorrelation, HR = {result:.1f}")
                return features
            
            # Process peaks if we found at least 2
            # Calculate RR intervals (in seconds)
            rr_intervals = np.diff(peaks) / self.fs
            
            # Try to filter physiologically implausible intervals
            valid_rr = (rr_intervals >= self.min_beat_interval_sec * 0.8) & (rr_intervals <= self.max_beat_interval_sec * 1.2)
            
            # If all intervals were filtered out, use the raw intervals
            if np.sum(valid_rr) == 0:
                if self.debug:
                    print(f"  PPG Feature Extraction: All RR intervals outside physiological range, using raw values")
                valid_rr = np.ones_like(rr_intervals, dtype=bool)
            
            # Get valid RR intervals
            valid_rr_intervals = rr_intervals[valid_rr]
            
            # Calculate heart rate even with just 1 interval
            if len(valid_rr_intervals) >= 1:
                features['heart_rate'] = 60 / np.mean(valid_rr_intervals)
            
            # If only few peaks, bail out early with just heart rate
            if len(peaks) < 4 or len(valid_rr_intervals) < 3:
                return features
            
            # Calculate time domain metrics if we have enough intervals
            features['sdnn'] = np.std(valid_rr_intervals)
            
            # RMSSD requires at least 2 successive differences (3 intervals)
            if len(valid_rr_intervals) >= 3:
                rmssd = np.sqrt(np.mean(np.diff(valid_rr_intervals)**2))
                features['rmssd'] = rmssd
                
                # pNN50 - Percentage of successive intervals differing by > 50ms
                differences = np.abs(np.diff(valid_rr_intervals))
                features['pnn50'] = np.sum(differences > 0.05) / len(differences)
            
            # Frequency domain measures require at least 4 intervals
            if len(valid_rr_intervals) >= 4:
                # Only attempt frequency domain if we have enough intervals
                try:
                    # Interpolate RR intervals to get evenly sampled time series
                    rr_times = np.cumsum(valid_rr_intervals)
                    rr_times = np.insert(rr_times, 0, 0)  # Add time 0
                    
                    # Create regular time axis (4 Hz is standard for HRV)
                    fs_interp = 4.0
                    max_time = rr_times[-1]  
                    
                    # Need reasonable length for interpolation
                    if max_time >= 2.0:  # At least 2 seconds of data
                        regular_times = np.arange(0, max_time, 1/fs_interp)
                        
                        if len(regular_times) >= 8:  # Need enough points for spectral analysis
                            try:
                                # Cubic spline interpolation
                                from scipy.interpolate import CubicSpline
                                cs = CubicSpline(rr_times[:-1], valid_rr_intervals)
                                rr_interpolated = cs(regular_times)
                            except:
                                # Fall back to linear interpolation if cubic fails
                                rr_interpolated = np.interp(regular_times, rr_times[:-1], valid_rr_intervals)
                            
                            # Ensure enough data points for welch
                            nperseg = min(len(rr_interpolated), 8)
                            if nperseg >= 4:  # Need at least 4 points for Welch
                                # Calculate power spectrum
                                f, psd = signal.welch(rr_interpolated, fs=fs_interp, 
                                                    nperseg=nperseg, 
                                                    detrend='constant', scaling='density')
                                
                                # LF (0.04-0.15 Hz) - sympathetic and parasympathetic activity
                                lf_band = (f >= 0.04) & (f <= 0.15)
                                if np.any(lf_band):
                                    features['lf_power'] = np.trapz(psd[lf_band], f[lf_band])
                                
                                # HF (0.15-0.4 Hz) - parasympathetic activity
                                hf_band = (f >= 0.15) & (f <= 0.4)
                                if np.any(hf_band):
                                    features['hf_power'] = np.trapz(psd[hf_band], f[hf_band])
                                
                                # LF/HF ratio
                                if not np.isnan(features['hf_power']) and features['hf_power'] > 0:
                                    features['lf_hf_ratio'] = features['lf_power'] / features['hf_power']
                except Exception as e:
                    if self.debug:
                        print(f"  PPG Feature Extraction: Frequency domain error: {e}")
                
        except Exception as e:
            if self.debug:
                print(f"  PPG Feature Extraction error: {e}")
            
        return features
        
    def estimate_heart_rate_via_autocorrelation(self, signal_data):
        """
        Estimate heart rate using signal autocorrelation when peak detection fails
        """
        if len(signal_data) < self.fs * 2:  # Need at least 2 seconds
            return None
            
        try:
            # Calculate autocorrelation
            correlation = np.correlate(signal_data, signal_data, mode='full')
            correlation = correlation[len(correlation)//2:]  # Use only positive lags
            
            # Find peaks in autocorrelation
            min_lag = int(self.fs * self.min_beat_interval_sec)  # Min physiological beat duration
            max_lag = int(self.fs * self.max_beat_interval_sec)  # Max physiological beat duration
            
            if min_lag >= len(correlation) or min_lag < 2:
                return None
                
            # Look for first peak after minimum lag
            peaks, _ = signal.find_peaks(correlation[min_lag:max_lag])
            
            if len(peaks) == 0:
                return None
                
            # First peak is our best estimate of cycle length
            cycle_samples = peaks[0] + min_lag
            beat_duration_sec = cycle_samples / self.fs
            
            # Convert to BPM
            estimated_hr = 60.0 / beat_duration_sec
            
            # Ensure it's within physiological range
            if 30 <= estimated_hr <= 180:
                return estimated_hr
            else:
                return None
                
        except Exception as e:
            if self.debug:
                print(f"  PPG Feature Extraction: Autocorrelation error: {e}")
            return None
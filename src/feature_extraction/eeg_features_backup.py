# src/feature_extraction/eeg_features.py
import numpy as np
import warnings
from scipy import signal
# Use absolute import from src
from src.utils.config_manager import load_config
from src.utils.signal_utils import (calculate_psd_welch, calculate_band_power, 
                                    calculate_spectral_edge_frequency)

# Try to import MNE and MNE-Connectivity for advanced features
# If not available, we'll still provide basic features
try:
    import mne
    from mne_connectivity import spectral_connectivity_epochs
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False
    warnings.warn("MNE and MNE-Connectivity not available. Using direct connectivity calculation.")

class EEGFeatureExtractor:
    def __init__(self):
        config = load_config()
        self.eeg_settings = config['signal_processing']['eeg']
        
        self.fs = self.eeg_settings['sample_rate']
        self.window_sec = self.eeg_settings['feature_window_sec'] 
        self.window_samples = int(self.window_sec * self.fs)
        
        self.psd_nperseg = int(self.eeg_settings['psd_nperseg_sec'] * self.fs)
        self.psd_noverlap = int(self.psd_nperseg * self.eeg_settings.get('psd_overlap_ratio', 0.5))

        # Define frequency bands
        self.theta_band = tuple(self.eeg_settings['theta_band_actual'])
        self.alpha_band = tuple(self.eeg_settings.get('alpha_band', [8.0, 13.0]))
        # Add additional bands for enhanced feature set
        self.delta_band = tuple(self.eeg_settings.get('delta_band', [1.0, 4.0]))
        
        self.total_power_ref_band = tuple(self.eeg_settings['total_power_reference_band'])
        self.sef_percentage = self.eeg_settings.get('sef_percentage', 0.95)

        self.channel_names = self.eeg_settings['active_channels_names'] 
        self.num_active_channels = len(self.channel_names)
        
        # Control connectivity calculation
        self.calculate_connectivity = True  # Set to False to disable connectivity features
        
        # Initialize feature names list
        self.feature_names = []
        
        # Add core spectral power features
        for ch_name in self.channel_names:
            # Core features from original implementation
            self.feature_names.extend([
                f"{ch_name}_abs_theta_power",
                f"{ch_name}_rel_theta_power",
                f"{ch_name}_sef{int(self.sef_percentage*100)}",
                f"{ch_name}_abs_alpha_power"
            ])
            
            # Add extended band powers
            self.feature_names.extend([
                f"{ch_name}_abs_delta_power",
                f"{ch_name}_rel_delta_power",
                f"{ch_name}_rel_alpha_power"
            ])
            
            # Add band ratios
            self.feature_names.extend([
                f"{ch_name}_alpha_theta_ratio"  # Alpha/Theta - higher in relaxed states
            ])
        
        # Add connectivity features if enabled
        if self.calculate_connectivity:
            self.connectivity_pairs = []
            for i in range(self.num_active_channels):
                for j in range(i+1, self.num_active_channels):
                    self.connectivity_pairs.append((i, j))
            
            # Add coherence and PLV features for each channel pair and frequency band
            for band_name in ['theta', 'alpha']:
                for pair in self.connectivity_pairs:
                    ch1, ch2 = self.channel_names[pair[0]], self.channel_names[pair[1]]
                    self.feature_names.extend([
                        f"{ch1}_{ch2}_{band_name}_coherence",
                        f"{ch1}_{ch2}_{band_name}_plv"
                    ])
        else:
            self.connectivity_pairs = []
        
        print(f"EEGFeatureExtractor Initialized for {self.num_active_channels} channels: {self.channel_names}")
        print(f"  Total features: {len(self.feature_names)}")
        if self.calculate_connectivity:
            print(f"  Connectivity computation: {'MNE + Direct' if MNE_AVAILABLE else 'Direct only'}")
        else:
            print(f"  Connectivity computation: Disabled")

    def calculate_direct_coherence(self, x, y, fs, band=None):
        """
        Calculate coherence between two signals using scipy.signal.coherence
        
        Parameters:
        -----------
        x, y : ndarray
            Input signals
        fs : float
            Sampling frequency
        band : tuple, optional
            Frequency band limits (low, high)
            
        Returns:
        --------
        float
            Mean coherence in the specified band
        """
        try:
            # Use a reasonable window size for coherence calculation
            nperseg = min(256, len(x) // 4, len(y) // 4)
            if nperseg < 16:  # Need minimum samples for coherence
                return np.nan
                
            f, Cxy = signal.coherence(x, y, fs, nperseg=nperseg)
            
            if band is None:
                return np.mean(Cxy)
            else:
                band_mask = (f >= band[0]) & (f <= band[1])
                if not np.any(band_mask):
                    return np.nan
                return np.mean(Cxy[band_mask])
        except Exception as e:
            return np.nan
            
    def calculate_phase_locking_value(self, x, y, fs, band=None):
        """
        Calculate phase locking value between two signals using Hilbert transform
        
        Parameters:
        -----------
        x, y : ndarray
            Input signals
        fs : float
            Sampling frequency
        band : tuple, optional
            Frequency band limits (low, high)
            
        Returns:
        --------
        float
            PLV in the specified band
        """
        try:
            # Filter signals if band is specified
            if band is not None:
                # Ensure we have enough samples for a good filter
                if len(x) < fs // 2:  # Need at least 0.5 seconds of data
                    return np.nan
                    
                # Design bandpass filter with reasonable parameters
                nyquist = fs / 2
                low = max(band[0] / nyquist, 0.01)  # Avoid DC
                high = min(band[1] / nyquist, 0.99)  # Avoid Nyquist
                
                if low >= high:
                    return np.nan
                    
                # Use lower order filter for short epochs
                b, a = signal.butter(2, [low, high], btype='band')
                
                # Apply filter
                x_filt = signal.filtfilt(b, a, x)
                y_filt = signal.filtfilt(b, a, y)
            else:
                x_filt = x
                y_filt = y
                
            # Apply Hilbert transform to get instantaneous phase
            x_phase = np.angle(signal.hilbert(x_filt))
            y_phase = np.angle(signal.hilbert(y_filt))
            
            # Calculate phase difference
            phase_diff = x_phase - y_phase
            
            # Calculate PLV
            plv = np.abs(np.mean(np.exp(1j * phase_diff)))
            return plv
        except Exception as e:
            return np.nan

    def calculate_mne_connectivity(self, data):
        """
        Calculate connectivity using MNE-connectivity (fixed version)
        
        Parameters:
        -----------
        data : ndarray
            EEG data (channels x samples)
            
        Returns:
        --------
        dict
            Dictionary of connectivity features
        """
        connectivity_features = {}
        
        if not MNE_AVAILABLE:
            return connectivity_features
            
        try:
            # Prepare data for MNE (needs to be epochs x channels x times)
            epochs_data = data.reshape(1, *data.shape)  # Add batch dimension
            
            # Define frequency bands
            freq_bands = {
                'theta': self.theta_band,
                'alpha': self.alpha_band
            }
            
            # Calculate connectivity for each band
            for band_name, (fmin, fmax) in freq_bands.items():
                try:
                    # Use spectral_connectivity_epochs (not spectral_connectivity_time)
                    con = spectral_connectivity_epochs(
                        epochs_data,
                        method=['coh', 'plv'],
                        mode='multitaper',
                        sfreq=self.fs,
                        fmin=fmin,
                        fmax=fmax,
                        faverage=True,  # Average across frequencies in band
                        mt_bandwidth=2,
                        n_jobs=1,
                        verbose=False
                    )
                    
                    # Extract connectivity matrices
                    coh_matrix = con[0].get_data()  # Coherence
                    plv_matrix = con[1].get_data()  # PLV
                    
                    # Store connectivity values for each channel pair
                    for pair_idx, pair in enumerate(self.connectivity_pairs):
                        ch1_idx, ch2_idx = pair
                        ch1_name = self.channel_names[ch1_idx]
                        ch2_name = self.channel_names[ch2_idx]
                        
                        # Extract values (shape: n_epochs x n_connections)
                        # Find the correct index in the connectivity matrix
                        conn_idx = None
                        for idx in range(coh_matrix.shape[1]):
                            # MNE connectivity typically uses upper triangle indexing
                            if idx == pair_idx:
                                conn_idx = idx
                                break
                        
                        if conn_idx is not None:
                            connectivity_features[f"{ch1_name}_{ch2_name}_{band_name}_coherence"] = coh_matrix[0, conn_idx]
                            connectivity_features[f"{ch1_name}_{ch2_name}_{band_name}_plv"] = plv_matrix[0, conn_idx]
                        else:
                            # Fallback: use the pair indices directly if available
                            if ch1_idx < coh_matrix.shape[1] and ch2_idx < coh_matrix.shape[2]:
                                connectivity_features[f"{ch1_name}_{ch2_name}_{band_name}_coherence"] = coh_matrix[0, ch1_idx, ch2_idx]
                                connectivity_features[f"{ch1_name}_{ch2_name}_{band_name}_plv"] = plv_matrix[0, ch1_idx, ch2_idx]
                
                except Exception as e:
                    # If MNE fails for this band, continue to next band
                    continue
                    
        except Exception as e:
            # If MNE completely fails, return empty dict (will use direct method)
            pass
            
        return connectivity_features

    def extract_features_from_epoch(self, preprocessed_eeg_epoch_active_channels):
        """Extract comprehensive EEG features from a single preprocessed epoch"""
        # Initialize features dictionary with NaNs
        features = {fname: np.nan for fname in self.feature_names}

        # Return early if epoch is None or all NaN
        if preprocessed_eeg_epoch_active_channels is None or \
            np.all(np.isnan(preprocessed_eeg_epoch_active_channels)):
            return features

        # Check shape
        if preprocessed_eeg_epoch_active_channels.shape != (self.num_active_channels, self.window_samples):
            return features

        # Calculate spectral features for each channel
        for i in range(self.num_active_channels):
            ch_name = self.channel_names[i]
            channel_epoch_data = preprocessed_eeg_epoch_active_channels[i, :]

            # Skip if channel contains NaN
            if np.any(np.isnan(channel_epoch_data)):
                continue 

            # Calculate PSD
            nperseg_calc = min(len(channel_epoch_data), self.psd_nperseg)
            noverlap_calc = int(nperseg_calc * self.eeg_settings.get('psd_overlap_ratio', 0.5)) if nperseg_calc > 0 else 0
            
            if nperseg_calc == 0: 
                freqs, psd = np.array([]), np.array([])
            else: 
                freqs, psd = calculate_psd_welch(channel_epoch_data, self.fs, 
                                            nperseg=nperseg_calc, noverlap=noverlap_calc)
            
            if psd.size == 0: 
                continue

            # Calculate total reference power for relative power calculations
            total_ref_pwr = calculate_band_power(freqs, psd, self.total_power_ref_band)

            # ---- Calculate absolute band powers ----
            abs_theta = calculate_band_power(freqs, psd, self.theta_band)
            features[f"{ch_name}_abs_theta_power"] = abs_theta
            
            abs_alpha = calculate_band_power(freqs, psd, self.alpha_band)
            features[f"{ch_name}_abs_alpha_power"] = abs_alpha
            
            abs_delta = calculate_band_power(freqs, psd, self.delta_band)
            features[f"{ch_name}_abs_delta_power"] = abs_delta
            
            # ---- Calculate relative band powers ----
            features[f"{ch_name}_rel_theta_power"] = calculate_band_power(freqs, psd, self.theta_band, 
                                                                    total_psd_power_in_ref_band=total_ref_pwr)
            features[f"{ch_name}_rel_delta_power"] = calculate_band_power(freqs, psd, self.delta_band, 
                                                                    total_psd_power_in_ref_band=total_ref_pwr)
            features[f"{ch_name}_rel_alpha_power"] = calculate_band_power(freqs, psd, self.alpha_band, 
                                                                    total_psd_power_in_ref_band=total_ref_pwr)
            
            # ---- Calculate band ratios ----
            if abs_theta > 0:
                features[f"{ch_name}_alpha_theta_ratio"] = abs_alpha / abs_theta
            
            # ---- Calculate spectral edge frequency ----
            sef_key = f"{ch_name}_sef{int(self.sef_percentage*100)}"
            features[sef_key] = calculate_spectral_edge_frequency(freqs, psd, total_ref_pwr, self.sef_percentage)
        
        # Calculate connectivity features if enabled
        if self.calculate_connectivity and len(self.connectivity_pairs) > 0:
            # First try MNE method (if available and no errors)
            mne_features = self.calculate_mne_connectivity(preprocessed_eeg_epoch_active_channels)
            
            # Use MNE results if available, otherwise use direct calculation
            for pair in self.connectivity_pairs:
                ch1_idx, ch2_idx = pair
                ch1_name = self.channel_names[ch1_idx]
                ch2_name = self.channel_names[ch2_idx]
                
                # Get data for both channels
                ch1_data = preprocessed_eeg_epoch_active_channels[ch1_idx, :]
                ch2_data = preprocessed_eeg_epoch_active_channels[ch2_idx, :]
                
                # Skip if either channel has NaN values
                if np.any(np.isnan(ch1_data)) or np.any(np.isnan(ch2_data)):
                    continue
                
                # For each frequency band
                for band_name, band_range in [('theta', self.theta_band), ('alpha', self.alpha_band)]:
                    coh_key = f"{ch1_name}_{ch2_name}_{band_name}_coherence"
                    plv_key = f"{ch1_name}_{ch2_name}_{band_name}_plv"
                    
                    # Use MNE result if available and valid
                    if coh_key in mne_features and not np.isnan(mne_features[coh_key]):
                        features[coh_key] = mne_features[coh_key]
                    else:
                        # Use direct calculation as fallback
                        features[coh_key] = self.calculate_direct_coherence(ch1_data, ch2_data, self.fs, band_range)
                    
                    if plv_key in mne_features and not np.isnan(mne_features[plv_key]):
                        features[plv_key] = mne_features[plv_key]
                    else:
                        # Use direct calculation as fallback
                        features[plv_key] = self.calculate_phase_locking_value(ch1_data, ch2_data, self.fs, band_range)
        
        return features
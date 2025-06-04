# realtime_processor.py - WITH COMPREHENSIVE DEBUG CODE

import numpy as np
import pandas as pd
import json
import time
import os
from collections import deque
import warnings

# Import the core processors and feature extractors
from src.data_processing.eeg_processor import EEGProcessor
from src.feature_extraction.eeg_features import EEGFeatureExtractor
from src.data_processing.ppg_processor import PPGProcessor
from src.feature_extraction.ppg_features import PPGFeatureExtractor
from src.data_processing.imu_processor import IMUProcessor
from src.feature_extraction.imu_features import IMUFeatureExtractor

class FixedNormalizationRealTimeProcessor:   
    def __init__(self, normalization_stats_path=None, session_buffer_duration_s=30):
        print("Initializing Real-Time Processor with Fixed Pre-trained Normalization...")
        
        # Initialize processors (matches offline)
        self.eeg_processor = EEGProcessor()
        self.eeg_feature_extractor = EEGFeatureExtractor()
        self.ppg_processor = PPGProcessor()
        self.ppg_feature_extractor = PPGFeatureExtractor()
        self.imu_processor = IMUProcessor()
        self.imu_feature_extractor = IMUFeatureExtractor()
        
        # Session context buffering (matches offline)
        self.session_buffer_duration_s = session_buffer_duration_s
        self.session_buffer_samples = int(session_buffer_duration_s * self.eeg_processor.fs)
        
        # Session buffers
        self.eeg_session_buffer = None
        self.sqc_session_buffer = None
        
        # FIXED normalization (inference mode - no adaptation)
        self.normalization_stats = {}
        self.normalization_loaded = False
        self.smoothing_buffers = {}  # For temporal smoothing only
        
        # Load pre-trained normalization statistics
        if normalization_stats_path and os.path.exists(normalization_stats_path):
            self.load_normalization_stats(normalization_stats_path)
        else:
            print("⚠️  No pre-trained normalization stats provided")
            print("   Features will be used without normalization")
        
        # Processing statistics
        self.processing_stats = {
            'total_epochs': 0,
            'standard_epochs': 0,
            'special_processed_epochs': 0,
            'software_quality_pass': 0,
            'hardware_sqc_available': 0,
            'hardware_sqc_fallback': 0,
            'normalization_applied': 0
        }
        
        print("✅ Real-Time Processor initialized:")
        print(f"   • EEG: {len(self.eeg_feature_extractor.feature_names)} features")
        print(f"   • Session buffer: {session_buffer_duration_s}s context")
        print(f"   • SQC handling: GRACEFUL (advisory, matches offline)")
        print(f"   • Normalization: FIXED pre-trained (inference mode)")
        print(f"   • Quality assessment: SOFTWARE-BASED (primary method)")
        print(f"   • Special processing: ENABLED for high-amplitude channels")
        
        # ADD DEBUG INFO
        print(f"\n🔍 DEBUG INFO:")
        print(f"   • EEG expected shape: ({self.eeg_feature_extractor.num_active_channels}, {self.eeg_feature_extractor.window_samples})")
        print(f"   • EEG channels: {self.eeg_feature_extractor.channel_names}")
        print(f"   • EEG sampling rate: {self.eeg_feature_extractor.fs} Hz")
        print(f"   • EEG window: {self.eeg_feature_extractor.window_sec} seconds")
    
    def load_normalization_stats(self, filepath):
        """Load PRE-TRAINED normalization statistics from offline processing"""
        try:
            with open(filepath, 'r') as f:
                loaded_data = json.load(f)
            
            self.normalization_stats = loaded_data.get('feature_stats', {})
            
            print(f"✅ Loaded FIXED normalization statistics:")
            print(f"   • Source: {filepath}")
            print(f"   • Features: {len(self.normalization_stats)}")
            print(f"   • Mode: INFERENCE (no adaptation during real-time)")
            
            # Verify we have the key features we need
            key_features = ['lf_rel_theta_power', 'rf_rel_alpha_power', 'lf_rel_alpha_power']
            available_key_features = [f for f in key_features if f in self.normalization_stats]
            print(f"   • Key features available: {len(available_key_features)}/{len(key_features)}")
            
            self.normalization_loaded = True
            return True
            
        except Exception as e:
            print(f"❌ Error loading normalization stats: {e}")
            print(f"   Continuing without normalization")
            self.normalization_loaded = False
            return False
    
    def _apply_fixed_normalization(self, features_dict_from_extractor):
        # features_dict_from_extractor might have mixed-case keys (e.g., LF_..., PPG_HR)
        
        if not self.normalization_loaded or not self.normalization_stats:
            print("[PROC DEBUG] Normalization skipped: Stats not loaded or empty.")
            # Ensure all keys in the returned dict are lowercase, even if no normalization is applied.
            return {k.lower(): v for k, v in features_dict_from_extractor.items()}

        # Start with a new dictionary where all original keys are converted to lowercase.
        # This ensures that features not explicitly processed by this normalization step
        # are still passed through with lowercase keys.
        processed_features_output = {k.lower(): v for k, v in features_dict_from_extractor.items()}
        
        current_norm_applied_count = 0
        current_smoothed_created_count = 0
        
        # Keywords and prefixes should be lowercase for comparison with feature_name_lower
        power_feature_keywords = ['_abs_', '_rel_', '_power', '_ratio']
        eeg_channel_prefixes = ['lf_', 'rf_', 'otel_', 'oter_'] 

        normalization_log_examples = [] # To store a few examples of normalization

        # Iterate based on the keys that are now in processed_features_output (all lowercase)
        for feature_name_lower, value in processed_features_output.items():
            
            # Preserve non-numeric types or NaNs as they are; FeatureAligner will handle NaNs.
            if not isinstance(value, (int, float, np.number)) or pd.isna(value):
                # If it's NaN, it's already stored as NaN with a lowercase key.
                # If non-numeric, it's also stored with a lowercase key.
                continue 

            # Determine if it's an EEG power feature eligible for normalization by this method
            is_eeg_power_feature = any(feature_name_lower.startswith(prefix) for prefix in eeg_channel_prefixes) and \
                                   any(keyword in feature_name_lower for keyword in power_feature_keywords)

            if is_eeg_power_feature:
                if feature_name_lower in self.normalization_stats:
                    stats = self.normalization_stats[feature_name_lower]
                    if (stats.get('initialized', False) and
                        isinstance(stats.get('mean'), (int, float)) and
                        isinstance(stats.get('std'), (int, float)) and
                        stats['std'] > 1e-8): # Check for valid std
                        
                        original_value_for_log = float(value) # Store before normalization for logging
                        normalized_value = (float(value) - stats['mean']) / stats['std']
                        processed_features_output[feature_name_lower] = normalized_value
                        current_norm_applied_count += 1

                        if len(normalization_log_examples) < 3: # Log first few examples
                            normalization_log_examples.append(
                                f"'{feature_name_lower}': {original_value_for_log:.3f} -> {normalized_value:.3f}"
                            )

                        # Create smoothed versions of *these normalized* alpha/theta features
                        if any(band_keyword in feature_name_lower for band_keyword in ['theta', 'alpha']):
                            # Construct the name for the smoothed feature, e.g., "lf_rel_theta_power_smoothed"
                            smoothed_feature_name_key = f"{feature_name_lower}_smoothed"
                            
                            # _apply_temporal_smoothing uses this key for its internal buffer
                            smoothed_value = self._apply_temporal_smoothing(smoothed_feature_name_key, normalized_value)
                            processed_features_output[smoothed_feature_name_key] = smoothed_value
                            current_smoothed_created_count += 1
                    else:
                        # Stats not initialized, or std is invalid. Keep original value (already float).
                        # print(f"[PROC DEBUG] Stats for '{feature_name_lower}' not initialized or std invalid. Original value {float(value):.3f} kept.")
                        processed_features_output[feature_name_lower] = float(value)
                else:
                    # Feature is an EEG power type, but no stats for it in ml_normalization_stats.json
                    # Keep original value (already float).
                    # print(f"[PROC DEBUG] No normalization stats for EEG power feature '{feature_name_lower}'. Original value {float(value):.3f} kept.")
                    processed_features_output[feature_name_lower] = float(value)
            else:
                # Not an EEG power feature this method normalizes.
                # Ensure it's a float if it was numeric. It's already in processed_features_output with a lowercase key.
                if isinstance(value, (int, np.integer)): # Should already be float if it passed the first check
                     processed_features_output[feature_name_lower] = float(value)


        if normalization_log_examples:
            print(f"[PROC DETAIL] Normalization applied for {current_norm_applied_count} EEG power features. Examples: {normalization_log_examples}")
        elif current_norm_applied_count > 0 : # No examples logged but count > 0
            print(f"[PROC DETAIL] Normalization applied for {current_norm_applied_count} EEG power features.")


        if current_smoothed_created_count > 0:
            print(f"[PROC DETAIL] Created {current_smoothed_created_count} smoothed features from normalized values.")

        self.processing_stats['normalization_applied_count_current_epoch'] = current_norm_applied_count
        self.processing_stats['smoothed_features_created_current_epoch'] = current_smoothed_created_count
        
        return processed_features_output
    
    def _apply_temporal_smoothing(self, feature_name, value):
        """Apply causal temporal smoothing (same as offline)"""
        if np.isnan(value):
            return value
        
        smoothing_window = 3  # Same as offline
        
        # Initialize buffer
        if feature_name not in self.smoothing_buffers:
            self.smoothing_buffers[feature_name] = []
        
        buffer = self.smoothing_buffers[feature_name]
        buffer.append(value)
        
        # Keep only last N values
        if len(buffer) > smoothing_window:
            buffer.pop(0)
        
        # Return smoothed value
        valid_values = [v for v in buffer if not np.isnan(v)]
        return np.mean(valid_values) if valid_values else value
    
    def _update_session_buffer(self, new_eeg_data, new_sqc_data=None):
        # Initialize or update EEG session buffer
        if self.eeg_session_buffer is None:
            self.eeg_session_buffer = new_eeg_data.copy()
            print(f"🔄 Initialized EEG session buffer: {self.eeg_session_buffer.shape}")
        else:
            self.eeg_session_buffer = np.concatenate([self.eeg_session_buffer, new_eeg_data], axis=1)
            
            # Keep only the last N samples (session context window)
            if self.eeg_session_buffer.shape[1] > self.session_buffer_samples:
                start_idx = self.eeg_session_buffer.shape[1] - self.session_buffer_samples
                self.eeg_session_buffer = self.eeg_session_buffer[:, start_idx:]
        
        # Handle SQC data
        if new_sqc_data is not None and len(new_sqc_data) == self.eeg_processor.num_active_channels:
            try:
                if self.sqc_session_buffer is None:
                    if isinstance(new_sqc_data, list):
                        new_sqc_data = np.array(new_sqc_data)
                    
                    if new_sqc_data.ndim == 1:
                        self.sqc_session_buffer = new_sqc_data.reshape(-1, 1)
                    else:
                        self.sqc_session_buffer = new_sqc_data.copy()
                
                self.processing_stats['hardware_sqc_available'] += 1
                
            except Exception as e:
                print(f"   ⚠️  SQC processing warning (continuing gracefully): {e}")
                self.sqc_session_buffer = None
                self.processing_stats['hardware_sqc_fallback'] += 1
        else:
            self.sqc_session_buffer = None
            self.processing_stats['hardware_sqc_fallback'] += 1
    
    def _apply_graceful_quality_assessment(self, processed_epoch, sqc_scores_for_epoch=None):
        """
        Apply graceful quality assessment (matches offline processing)
        Same logic as offline: software primary, SQC advisory
        """
        quality_info = {
            'method': 'software_primary_fixed_norm',
            'hardware_sqc_available': sqc_scores_for_epoch is not None,
            'hardware_sqc_scores': sqc_scores_for_epoch.tolist() if sqc_scores_for_epoch is not None else None,
            'normalization_mode': 'fixed_pretrained',
            'software_quality_metrics': {}
        }
        
        # Software-based quality assessment
        try:
            # Calculate quality metrics from processed signal
            epoch_std = np.std(processed_epoch, axis=1)
            epoch_mean = np.mean(np.abs(processed_epoch), axis=1)
            epoch_range = np.ptp(processed_epoch, axis=1)
            
            # Same artifact detection
            artifact_threshold = 0.15  # 15% artifact threshold
            
            artifact_flags = []
            for ch_idx, ch_name in enumerate(self.eeg_processor.active_channel_names):
                # Same special processing logic as offline
                is_high_amplitude_channel = ch_name in ['RF', 'OTER']
                
                # Same adaptive thresholds as offline
                if is_high_amplitude_channel:
                    std_threshold = 80.0  # More lenient for high-amplitude channels
                    range_threshold = 400.0
                else:
                    std_threshold = 50.0  # Standard thresholds
                    range_threshold = 250.0
                
                # Same artifact detection logic as offline
                excessive_noise = epoch_std[ch_idx] > std_threshold
                excessive_range = epoch_range[ch_idx] > range_threshold
                flat_signal = epoch_std[ch_idx] < 0.1
                
                channel_artifact = excessive_noise or excessive_range or flat_signal
                artifact_flags.append(channel_artifact)
                
                quality_info['software_quality_metrics'][f'{ch_name}_std'] = epoch_std[ch_idx]
                quality_info['software_quality_metrics'][f'{ch_name}_range'] = epoch_range[ch_idx]
                quality_info['software_quality_metrics'][f'{ch_name}_artifact'] = channel_artifact
            
            # Same overall quality decision as offline
            artifact_rate = np.mean(artifact_flags)
            software_quality_pass = artifact_rate < artifact_threshold
            
            quality_info['software_artifact_rate'] = artifact_rate
            quality_info['software_quality_pass'] = software_quality_pass
            
        except Exception as e:
            print(f"   ⚠️  Software quality assessment error: {e}")
            software_quality_pass = False
            quality_info['software_error'] = str(e)
        
        # Hardware SQC as additional information
        hardware_sqc_pass = True  # Default to pass if no SQC
        if sqc_scores_for_epoch is not None:
            try:
                good_channels = np.sum(sqc_scores_for_epoch >= 1)
                total_channels = len(sqc_scores_for_epoch)
                hardware_sqc_pass = good_channels >= 1
                
                quality_info['hardware_good_channels'] = good_channels
                quality_info['hardware_total_channels'] = total_channels
                quality_info['hardware_sqc_pass'] = hardware_sqc_pass
                
            except Exception as e:
                print(f"   ⚠️  Hardware SQC error (ignoring gracefully): {e}")
                hardware_sqc_pass = True
        
        # Final quality decision
        if software_quality_pass:
            final_quality = True
            quality_info['decision_reason'] = 'software_quality_pass'
        elif hardware_sqc_pass and sqc_scores_for_epoch is not None:
            final_quality = True
            quality_info['decision_reason'] = 'hardware_sqc_override'
        else:
            final_quality = False
            quality_info['decision_reason'] = 'both_failed_or_software_only_failed'
        
        # Check if this required special processing
        is_special_processed = any(ch in ['RF', 'OTER'] for ch in self.eeg_processor.active_channel_names)
        
        return final_quality, is_special_processed, quality_info
    
    def _debug_eeg_feature_extraction(self, processed_eeg_epoch, eeg_features, epoch_count):
        """
        Comprehensive debug logging for EEG feature extraction
        """
        print(f"\n🔍 === EEG FEATURE EXTRACTION DEBUG (Epoch {epoch_count}) ===")
        
        # Step 1: Input validation
        print(f"[DEBUG-1] INPUT VALIDATION:")
        if processed_eeg_epoch is None:
            print(f"   ❌ Processed epoch is None!")
            return
        
        expected_shape = (self.eeg_feature_extractor.num_active_channels, self.eeg_feature_extractor.window_samples)
        actual_shape = processed_eeg_epoch.shape
        
        print(f"   Expected shape: {expected_shape}")
        print(f"   Actual shape: {actual_shape}")
        print(f"   Channels: {self.eeg_feature_extractor.channel_names}")
        print(f"   Window samples: {self.eeg_feature_extractor.window_samples}")
        print(f"   Sampling rate: {self.eeg_feature_extractor.fs} Hz")
        
        # Step 2: Shape validation
        print(f"\n[DEBUG-2] SHAPE VALIDATION:")
        if actual_shape != expected_shape:
            print(f"   ❌ SHAPE MISMATCH! This causes all NaN features!")
            print(f"   Expected: {expected_shape}")
            print(f"   Got: {actual_shape}")
            print(f"   Difference: channels={actual_shape[0]-expected_shape[0]}, samples={actual_shape[1]-expected_shape[1]}")
        else:
            print(f"   ✅ Shape validation passed")
        
        # Step 3: Data quality check
        print(f"\n[DEBUG-3] DATA QUALITY CHECK:")
        has_nan = np.any(np.isnan(processed_eeg_epoch))
        has_inf = np.any(np.isinf(processed_eeg_epoch))
        
        print(f"   Has NaN: {has_nan}")
        print(f"   Has Inf: {has_inf}")
        
        if not has_nan and not has_inf:
            data_min = processed_eeg_epoch.min()
            data_max = processed_eeg_epoch.max()
            data_mean = processed_eeg_epoch.mean()
            data_std = processed_eeg_epoch.std()
            
            print(f"   Data range: [{data_min:.3f}, {data_max:.3f}]")
            print(f"   Data mean: {data_mean:.3f}")
            print(f"   Data std: {data_std:.3f}")
            
            # Check each channel
            for i, ch_name in enumerate(self.eeg_feature_extractor.channel_names):
                if i < actual_shape[0]:
                    ch_data = processed_eeg_epoch[i, :]
                    ch_has_nan = np.any(np.isnan(ch_data))
                    ch_range = [ch_data.min(), ch_data.max()] if not ch_has_nan else ['NaN', 'NaN']
                    ch_std = ch_data.std() if not ch_has_nan else 'NaN'
                    
                    status = "❌ NaN" if ch_has_nan else "✅"
                    print(f"   {ch_name}: Range [{ch_range[0]:.3f}, {ch_range[1]:.3f}], Std: {ch_std:.3f} {status}")
        else:
            print(f"   ❌ Data contains NaN or Inf values!")
        
        # Step 4: Feature extraction results
        print(f"\n[DEBUG-4] FEATURE EXTRACTION RESULTS:")
        if eeg_features:
            total_features = len(eeg_features)
            nan_features = sum(1 for v in eeg_features.values() if pd.isna(v))
            valid_features = total_features - nan_features
            
            print(f"   Total features: {total_features}")
            print(f"   Valid features: {valid_features}")
            print(f"   NaN features: {nan_features}")
            print(f"   Success rate: {valid_features/total_features*100:.1f}%")
            
            if valid_features > 0:
                print(f"   ✅ Some features extracted successfully!")
                # Show sample valid features
                valid_samples = [(k, v) for k, v in eeg_features.items() if not pd.isna(v)][:3]
                for name, value in valid_samples:
                    print(f"      {name}: {value:.6f}")
            else:
                print(f"   ❌ ALL FEATURES ARE NaN!")
                
                # Show sample feature names that failed
                sample_failed = list(eeg_features.keys())[:5]
                print(f"   Sample failed features: {sample_failed}")
        else:
            print(f"   ❌ No features returned!")
        
        # Step 5: Test with single channel if possible
        if not has_nan and not has_inf and actual_shape == expected_shape:
            print(f"\n[DEBUG-5] SINGLE CHANNEL TEST:")
            try:
                from src.utils.signal_utils import calculate_psd_welch, calculate_band_power
                
                # Test with first channel
                test_channel = processed_eeg_epoch[0, :]
                test_ch_name = self.eeg_feature_extractor.channel_names[0]
                
                print(f"   Testing PSD calculation for {test_ch_name}...")
                
                # Calculate PSD
                nperseg_calc = min(len(test_channel), self.eeg_feature_extractor.psd_nperseg)
                noverlap_calc = int(nperseg_calc * 0.5)
                
                print(f"   nperseg: {nperseg_calc}, noverlap: {noverlap_calc}")
                
                freqs, psd = calculate_psd_welch(
                    test_channel, 
                    self.eeg_feature_extractor.fs, 
                    nperseg=nperseg_calc, 
                    noverlap=noverlap_calc
                )
                
                print(f"   PSD calculated: {len(freqs)} frequency bins")
                print(f"   Frequency range: [{freqs.min():.2f}, {freqs.max():.2f}] Hz")
                print(f"   PSD range: [{psd.min():.6f}, {psd.max():.6f}]")
                
                # Test band power calculation
                theta_band = self.eeg_feature_extractor.theta_band
                total_ref_band = self.eeg_feature_extractor.total_power_ref_band
                
                total_ref_pwr = calculate_band_power(freqs, psd, total_ref_band)
                abs_theta = calculate_band_power(freqs, psd, theta_band)
                
                print(f"   Total ref power: {total_ref_pwr:.6f}")
                print(f"   Abs theta power: {abs_theta:.6f}")
                
                if pd.isna(total_ref_pwr) or pd.isna(abs_theta):
                    print(f"   ❌ Band power calculation returned NaN!")
                else:
                    print(f"   ✅ Band power calculation successful!")
                    
            except Exception as e:
                print(f"   ❌ Single channel test failed: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"🔍 === END EEG FEATURE EXTRACTION DEBUG ===\n")
    
    def process_realtime_data(self, eeg_data, ppg_data=None, imu_data=None, sqc_scores=None, 
                            session_duration=0, streamer_scores=None):
        if eeg_data is None or eeg_data.size == 0:
            return []
        
        print(f" Processing with graceful SQC + fixed normalization")
        
        # Preprocess raw EEG data (same as offline format)
        if eeg_data.shape[1] == self.eeg_processor.total_device_channels:
            eeg_data = eeg_data.T
            print(f"   Transposed EEG data: {eeg_data.shape}")
        
        # Select active channels (same as offline)
        if eeg_data.shape[0] == self.eeg_processor.total_device_channels:
            active_indices = self.eeg_processor.active_channel_indices
            eeg_data_active = eeg_data[active_indices, :]
            print(f"   Selected active channels: {eeg_data_active.shape}")
        else:
            eeg_data_active = eeg_data
        
        # Update session buffer (maintains preprocessing context)
        self._update_session_buffer(eeg_data_active, sqc_scores)
        
        # Check session buffer readiness
        if (self.eeg_session_buffer is None or 
            self.eeg_session_buffer.shape[1] < self.eeg_processor.window_samples * 2):
            print(f"   Session buffer building... need more context")
            return []
        
        print(f"✅ Session buffer ready: {self.eeg_session_buffer.shape[1]} samples")
        
        # Generate epochs with SQC
        extracted_features = []
        
        try:
            epoch_count = 0
            for processed_eeg_epoch, original_quality_flag, epoch_start_s, special_processed in \
                self.eeg_processor.generate_preprocessed_epochs(
                    self.eeg_session_buffer, 
                    self.sqc_session_buffer
                ):
                
                epoch_count += 1
                self.processing_stats['total_epochs'] += 1
                
                # Apply graceful quality assessment
                sqc_scores_for_epoch = None
                if self.sqc_session_buffer is not None and self.sqc_session_buffer.shape[1] > 0:
                    sqc_timepoint_idx = min(epoch_count - 1, self.sqc_session_buffer.shape[1] - 1)
                    sqc_scores_for_epoch = self.sqc_session_buffer[:, sqc_timepoint_idx]
                
                # Graceful quality assessment
                final_quality, is_special_processed, quality_info = self._apply_graceful_quality_assessment(
                    processed_eeg_epoch, sqc_scores_for_epoch
                )
                
                # Update statistics
                if special_processed or is_special_processed:
                    self.processing_stats['special_processed_epochs'] += 1
                else:
                    self.processing_stats['standard_epochs'] += 1
                
                if final_quality:
                    self.processing_stats['software_quality_pass'] += 1
                
                print(f"    📊 Epoch {epoch_count}: Quality {'PASS' if final_quality else 'FAIL'} "
                      f"({'Special' if special_processed else 'Standard'})")
                print(f"        Reason: {quality_info['decision_reason']}")
                
                # Extract features for quality-passed epochs
                if final_quality and processed_eeg_epoch is not None:
                    
                    # ========== ADD COMPREHENSIVE DEBUG CODE HERE ==========
                    print(f"\n🔍 === PRE-EXTRACTION DEBUG (Epoch {epoch_count}) ===")
                    print(f"[PRE-DEBUG] About to extract EEG features...")
                    print(f"[PRE-DEBUG] Processed epoch shape: {processed_eeg_epoch.shape}")
                    print(f"[PRE-DEBUG] Expected shape: ({self.eeg_feature_extractor.num_active_channels}, {self.eeg_feature_extractor.window_samples})")
                    print(f"[PRE-DEBUG] Quality flag: {final_quality}")
                    print(f"[PRE-DEBUG] Special processing: {special_processed or is_special_processed}")
                    
                    # Extract EEG features
                    eeg_features = self.eeg_feature_extractor.extract_features_from_epoch(processed_eeg_epoch)
                    
                    # ========== COMPREHENSIVE DEBUG AFTER EXTRACTION ==========
                    self._debug_eeg_feature_extraction(processed_eeg_epoch, eeg_features, epoch_count)
                    
                    # Quick summary of extraction results
                    if eeg_features:
                        nan_count = sum(1 for v in eeg_features.values() if pd.isna(v))
                        total_count = len(eeg_features)
                        valid_count = total_count - nan_count
                        
                        print(f"[EXTRACTION SUMMARY] Features: {valid_count}/{total_count} valid ({valid_count/total_count*100:.1f}%)")
                        
                        if nan_count == total_count:
                            print(f"[EXTRACTION SUMMARY] ❌ ALL FEATURES ARE NaN - CRITICAL ISSUE!")
                        elif valid_count > 0:
                            print(f"[EXTRACTION SUMMARY] ✅ Some features valid")
                            # Show first few valid features
                            valid_features = [(k, v) for k, v in eeg_features.items() if not pd.isna(v)][:3]
                            for name, value in valid_features:
                                print(f"[EXTRACTION SUMMARY]    {name}: {value:.6f}")
                    else:
                        print(f"[EXTRACTION SUMMARY] ❌ No features dictionary returned!")
                    
                    # ========== END DEBUG CODE ==========
                    
                    # Apply FIXED pre-trained normalization (inference mode)
                    normalized_features = self._apply_fixed_normalization(eeg_features)
                    
                    # Create epoch feature dictionary
                    epoch_features = {
                        'epoch_start_time_s': epoch_start_s,
                        'eeg_quality_flag': final_quality,
                        'special_processing': special_processed or is_special_processed,
                        'quality_assessment_method': quality_info['method'],
                        'quality_decision_reason': quality_info['decision_reason'],
                        'hardware_sqc_available': quality_info['hardware_sqc_available'],
                        'normalization_mode': 'fixed_pretrained',
                        'normalization_applied': self.normalization_loaded,
                        'session_duration_s': session_duration
                    }
                    
                    # Add normalized features
                    epoch_features.update(normalized_features)
                    
                    extracted_features.append(epoch_features)
                    
                # Limit epochs per call
                if epoch_count >= 3:
                    break
                    
        except Exception as e:
            print(f"❌ Error in processing: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"✅ Processing complete: {len(extracted_features)} quality epochs")
        print(f"   Normalization: {'Applied' if self.normalization_loaded else 'Not available'}")
        
        return extracted_features
    
    def get_processing_stats(self):
        """Get current processing statistics"""
        return self.processing_stats.copy()
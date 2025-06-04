# realtime_processor.py - FINAL FIX: HR TIME ALIGNMENT + NORMALIZATION LOADING

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
from src.data_processing.hr_processor import HRProcessor
from src.feature_extraction.hr_features import HRFeatureExtractor
from src.data_processing.imu_processor import IMUProcessor
from src.feature_extraction.imu_features import IMUFeatureExtractor
from streaming_utils import validate_window_shapes

class FixedNormalizationRealTimeProcessor:   
    def __init__(self, normalization_stats_path=None, session_buffer_duration_s=30):
        print("Initializing Real-Time Processor with Fixed Pre-trained Normalization...")
        
        # Initialize processors (matches offline)
        self.eeg_processor = EEGProcessor()
        
        # ========== REAL-TIME THRESHOLD OVERRIDE ==========
        # Override strict thresholds for real-time processing
        original_threshold = self.eeg_processor.epoch_artifact_threshold
        original_gating = self.eeg_processor.enable_session_level_gating
        
        # Set more lenient values for real-time
        self.eeg_processor.epoch_artifact_threshold = 50.0  # Allow up to 50% artifacts
        self.eeg_processor.enable_session_level_gating = False  # Disable strict gating
        
        print(f"\n⚠️  REAL-TIME THRESHOLD OVERRIDE:")
        print(f"   • Artifact threshold: {original_threshold}% → {self.eeg_processor.epoch_artifact_threshold}%")
        print(f"   • Session gating: {original_gating} → {self.eeg_processor.enable_session_level_gating}")
        print(f"   • Reason: Real-time data is noisier than offline recordings")
        # ========== END OF OVERRIDE ==========
        
        # Continue with rest of initialization
        self.eeg_feature_extractor = EEGFeatureExtractor()
        
        # HR processing
        self.hr_processor = HRProcessor()
        self.hr_feature_extractor = HRFeatureExtractor()
        
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
        
        # FIXED: Session timing tracking for HR alignment
        self.session_start_absolute_time = None
        self.last_session_duration = 0
        
        # Load pre-trained normalization statistics
        if normalization_stats_path and os.path.exists(normalization_stats_path):
            success = self.load_normalization_stats(normalization_stats_path)
            if not success:
                print("⚠️  Failed to load normalization stats - features will not be normalized")
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
        
        print("\n✅ Real-Time Processor initialized:")
        print(f"   • EEG: {len(self.eeg_feature_extractor.feature_names)} features")
        print(f"   • HR: {len(self.hr_feature_extractor.feature_names)} features (REPLACES PPG)")
        print(f"   • IMU: Available for movement features")
        print(f"   • Session buffer: {session_buffer_duration_s}s context")
        print(f"   • SQC handling: GRACEFUL (advisory, matches offline)")
        print(f"   • Normalization: {'LOADED' if self.normalization_loaded else 'NOT LOADED'}")
        print(f"   • Quality assessment: SOFTWARE-BASED (primary method)")
        print(f"   • Special processing: ENABLED for high-amplitude channels")
        print(f"   • Artifact threshold: {self.eeg_processor.epoch_artifact_threshold}% (overridden)")
    
    def load_normalization_stats(self, filepath):
        """Load PRE-TRAINED normalization statistics from offline processing - FINAL FIXED VERSION"""
        try:
            print(f"\n🔍 Loading normalization stats from: {filepath}")
            
            if not os.path.exists(filepath):
                print(f"❌ File does not exist: {filepath}")
                return False
            
            with open(filepath, 'r') as f:
                loaded_data = json.load(f)
            
            # FIXED: Handle the correct structure from your offline processing
            if 'feature_stats' in loaded_data:
                self.normalization_stats = loaded_data['feature_stats']
            else:
                # Fallback: maybe stats are at root level
                self.normalization_stats = loaded_data
            
            print(f"✅ Successfully loaded normalization statistics:")
            print(f"   • Source: {filepath}")
            print(f"   • Features: {len(self.normalization_stats)}")
            print(f"   • Mode: INFERENCE (no adaptation during real-time)")
            
            # FIXED: Verify we have the key features we need (try both cases)
            key_features_to_check = [
                ['lf_rel_theta_power', 'LF_rel_theta_power'],
                ['rf_rel_alpha_power', 'RF_rel_alpha_power'], 
                ['lf_rel_alpha_power', 'LF_rel_alpha_power']
            ]
            
            available_key_features = []
            for feature_variants in key_features_to_check:
                found = False
                for variant in feature_variants:
                    if variant in self.normalization_stats:
                        available_key_features.append(variant)
                        found = True
                        break
                if not found:
                    print(f"   • Key feature not found: {feature_variants}")
            
            print(f"   • Key features found: {available_key_features}")
            
            # Debug: Show first few feature names to understand the naming convention
            feature_names = list(self.normalization_stats.keys())[:5]
            print(f"   • Sample features: {feature_names}")
            
            # Verify feature structure
            if feature_names:
                sample_feature = feature_names[0]
                sample_stats = self.normalization_stats[sample_feature]
                required_keys = ['mean', 'std', 'initialized']
                has_required = all(key in sample_stats for key in required_keys)
                print(f"   • Feature structure valid: {has_required}")
                if has_required:
                    print(f"   • Sample: {sample_feature} = mean:{sample_stats['mean']:.3f}, std:{sample_stats['std']:.3f}")
            
            self.normalization_loaded = True
            return True
            
        except Exception as e:
            print(f"❌ Error loading normalization stats: {e}")
            print(f"   File exists: {os.path.exists(filepath)}")
            if os.path.exists(filepath):
                print(f"   File size: {os.path.getsize(filepath)} bytes")
            print(f"   Continuing without normalization")
            self.normalization_loaded = False
            return False
    
    def _apply_fixed_normalization(self, features_dict_from_extractor):
        """Apply fixed pre-trained normalization to features - FINAL FIXED VERSION"""
        if not self.normalization_loaded or not self.normalization_stats:
            print("[PROC DEBUG] Normalization skipped: Stats not loaded or empty.")
            return {k.lower(): v for k, v in features_dict_from_extractor.items()}

        print(f"[PROC DEBUG] Applying normalization to {len(features_dict_from_extractor)} features...")
        
        processed_features_output = {k.lower(): v for k, v in features_dict_from_extractor.items()}
        
        current_norm_applied_count = 0
        current_smoothed_created_count = 0
        
        power_feature_keywords = ['_abs_', '_rel_', '_power', '_ratio']
        eeg_channel_prefixes = ['lf_', 'rf_', 'otel_', 'oter_'] 
        normalization_log_examples = []
        
        # FIXED: Collect smoothed features separately to avoid "dictionary changed size during iteration"
        smoothed_features_to_add = {}

        # FIXED: Iterate over a copy of the items to avoid dictionary modification during iteration
        for feature_name_lower, value in list(processed_features_output.items()):
            if not isinstance(value, (int, float, np.number)) or pd.isna(value):
                continue 

            is_eeg_power_feature = any(feature_name_lower.startswith(prefix) for prefix in eeg_channel_prefixes) and \
                                   any(keyword in feature_name_lower for keyword in power_feature_keywords)

            if is_eeg_power_feature:
                # FIXED: Try multiple case variations to match normalization stats
                norm_key = None
                
                # Try different case variations
                possible_keys = [
                    feature_name_lower,                           # lf_rel_theta_power
                    feature_name_lower.upper(),                   # LF_REL_THETA_POWER  
                    feature_name_lower.replace('_', '_').title(), # Lf_Rel_Theta_Power
                    # Convert to proper case (first letter of each word uppercase)
                    '_'.join(word.capitalize() for word in feature_name_lower.split('_'))  # Lf_Rel_Theta_Power
                ]
                
                for possible_key in possible_keys:
                    if possible_key in self.normalization_stats:
                        norm_key = possible_key
                        break
                
                if norm_key:
                    stats = self.normalization_stats[norm_key]
                    if (stats.get('initialized', False) and
                        isinstance(stats.get('mean'), (int, float)) and
                        isinstance(stats.get('std'), (int, float)) and
                        stats['std'] > 1e-8):
                        
                        original_value_for_log = float(value)
                        normalized_value = (float(value) - stats['mean']) / stats['std']
                        processed_features_output[feature_name_lower] = normalized_value
                        current_norm_applied_count += 1

                        if len(normalization_log_examples) < 3:
                            normalization_log_examples.append(
                                f"'{feature_name_lower}': {original_value_for_log:.3f} -> {normalized_value:.3f} (using '{norm_key}')"
                            )

                        # FIXED: Create smoothed features for theta/alpha bands - store separately
                        if any(band_keyword in feature_name_lower for band_keyword in ['theta', 'alpha']):
                            smoothed_feature_name_key = f"{feature_name_lower}_smoothed"
                            smoothed_value = self._apply_temporal_smoothing(smoothed_feature_name_key, normalized_value)
                            smoothed_features_to_add[smoothed_feature_name_key] = smoothed_value
                            current_smoothed_created_count += 1
                    else:
                        processed_features_output[feature_name_lower] = float(value)
                else:
                    # No normalization stats found for this feature
                    processed_features_output[feature_name_lower] = float(value)
                    if current_norm_applied_count < 3:  # Only log first few missing features
                        print(f"[PROC DEBUG] No normalization stats found for '{feature_name_lower}'")
            else:
                if isinstance(value, (int, np.integer)):
                     processed_features_output[feature_name_lower] = float(value)

        # FIXED: Add smoothed features after iteration is complete
        processed_features_output.update(smoothed_features_to_add)

        if normalization_log_examples:
            print(f"[PROC DETAIL] Normalization applied for {current_norm_applied_count} EEG power features. Examples: {normalization_log_examples}")
        elif current_norm_applied_count > 0:
            print(f"[PROC DETAIL] Normalization applied for {current_norm_applied_count} EEG power features.")

        if current_smoothed_created_count > 0:
            print(f"[PROC DETAIL] Created {current_smoothed_created_count} smoothed features from normalized values.")

        self.processing_stats['normalization_applied'] += current_norm_applied_count
        
        return processed_features_output
    
    def _apply_temporal_smoothing(self, feature_name, value):
        """Apply causal temporal smoothing (same as offline)"""
        if np.isnan(value):
            return value
        
        smoothing_window = 3
        
        if feature_name not in self.smoothing_buffers:
            self.smoothing_buffers[feature_name] = []
        
        buffer = self.smoothing_buffers[feature_name]
        buffer.append(value)
        
        if len(buffer) > smoothing_window:
            buffer.pop(0)
        
        valid_values = [v for v in buffer if not np.isnan(v)]
        return np.mean(valid_values) if valid_values else value
    
    def _update_session_buffer(self, new_eeg_data, new_sqc_data=None):
        """Update session buffer for EEG and SQC data"""
        if self.eeg_session_buffer is None:
            self.eeg_session_buffer = new_eeg_data.copy()
            print(f"🔄 Initialized EEG session buffer: {self.eeg_session_buffer.shape}")
        else:
            self.eeg_session_buffer = np.concatenate([self.eeg_session_buffer, new_eeg_data], axis=1)
            
            if self.eeg_session_buffer.shape[1] > self.session_buffer_samples:
                start_idx = self.eeg_session_buffer.shape[1] - self.session_buffer_samples
                self.eeg_session_buffer = self.eeg_session_buffer[:, start_idx:]
        
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
        """Apply graceful quality assessment (matches offline processing)"""
        quality_info = {
            'method': 'software_primary_fixed_norm',
            'hardware_sqc_available': sqc_scores_for_epoch is not None,
            'hardware_sqc_scores': sqc_scores_for_epoch.tolist() if sqc_scores_for_epoch is not None else None,
            'normalization_mode': 'fixed_pretrained',
            'software_quality_metrics': {}
        }
        
        try:
            epoch_std = np.std(processed_epoch, axis=1)
            epoch_mean = np.mean(np.abs(processed_epoch), axis=1)
            epoch_range = np.ptp(processed_epoch, axis=1)
            
            artifact_threshold = 0.15
            
            artifact_flags = []
            for ch_idx, ch_name in enumerate(self.eeg_processor.active_channel_names):
                is_high_amplitude_channel = ch_name in ['RF', 'OTER']
                
                if is_high_amplitude_channel:
                    std_threshold = 80.0
                    range_threshold = 400.0
                else:
                    std_threshold = 50.0
                    range_threshold = 250.0
                
                excessive_noise = epoch_std[ch_idx] > std_threshold
                excessive_range = epoch_range[ch_idx] > range_threshold
                flat_signal = epoch_std[ch_idx] < 0.1
                
                channel_artifact = excessive_noise or excessive_range or flat_signal
                artifact_flags.append(channel_artifact)
                
                quality_info['software_quality_metrics'][f'{ch_name}_std'] = epoch_std[ch_idx]
                quality_info['software_quality_metrics'][f'{ch_name}_range'] = epoch_range[ch_idx]
                quality_info['software_quality_metrics'][f'{ch_name}_artifact'] = channel_artifact
            
            artifact_rate = np.mean(artifact_flags)
            software_quality_pass = artifact_rate < artifact_threshold
            
            quality_info['software_artifact_rate'] = artifact_rate
            quality_info['software_quality_pass'] = software_quality_pass
            
        except Exception as e:
            print(f"   ⚠️  Software quality assessment error: {e}")
            software_quality_pass = False
            quality_info['software_error'] = str(e)
        
        hardware_sqc_pass = True
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
        
        if software_quality_pass:
            final_quality = True
            quality_info['decision_reason'] = 'software_quality_pass'
        elif hardware_sqc_pass and sqc_scores_for_epoch is not None:
            final_quality = True
            quality_info['decision_reason'] = 'hardware_sqc_override'
        else:
            final_quality = False
            quality_info['decision_reason'] = 'both_failed_or_software_only_failed'
        
        is_special_processed = any(ch in ['RF', 'OTER'] for ch in self.eeg_processor.active_channel_names)
        
        return final_quality, is_special_processed, quality_info

    def _prepare_imu_epoch(self, imu_data, epoch_start_s):
        """Prepare IMU data epoch for feature extraction - convert to expected format"""
        try:
            if isinstance(imu_data, dict):
                accel_data = imu_data.get('accel', imu_data.get('accelerometer'))
                gyro_data = imu_data.get('gyro', imu_data.get('gyroscope'))
                
                if accel_data is not None:
                    if isinstance(accel_data, list):
                        accel_data = np.array(accel_data)
                    
                    expected_imu_samples = int(4.0 * getattr(self.imu_processor, 'fs', 125))
                    
                    if len(accel_data) >= expected_imu_samples:
                        # Prepare in the format expected by IMUFeatureExtractor: (6, samples)
                        # Channels 0-2: accel_x, accel_y, accel_z
                        # Channels 3-5: gyro_x, gyro_y, gyro_z
                        accel_epoch = accel_data[-expected_imu_samples:]
                        
                        if accel_epoch.ndim == 1:
                            # If 1D, assume it's magnitude - create 3D representation
                            imu_epoch = np.zeros((6, expected_imu_samples))
                            imu_epoch[0, :] = accel_epoch  # Put in accel_x
                        elif accel_epoch.shape[0] == 3:  # (3, samples) - x,y,z
                            imu_epoch = np.zeros((6, expected_imu_samples))
                            imu_epoch[0:3, :] = accel_epoch
                        elif accel_epoch.shape[1] == 3:  # (samples, 3) - transpose
                            imu_epoch = np.zeros((6, expected_imu_samples))
                            imu_epoch[0:3, :] = accel_epoch.T
                        else:
                            print(f"   Unexpected accel data shape: {accel_epoch.shape}")
                            return None
                        
                        # Add gyro data if available
                        if gyro_data is not None:
                            gyro_epoch = gyro_data[-expected_imu_samples:]
                            if isinstance(gyro_epoch, list):
                                gyro_epoch = np.array(gyro_epoch)
                            
                            if gyro_epoch.ndim == 1:
                                imu_epoch[3, :] = gyro_epoch  # Put in gyro_x
                            elif gyro_epoch.shape[0] == 3:
                                imu_epoch[3:6, :] = gyro_epoch
                            elif gyro_epoch.shape[1] == 3:
                                imu_epoch[3:6, :] = gyro_epoch.T
                        
                        if not np.any(np.isnan(imu_epoch)):
                            return imu_epoch
                        else:
                            print(f"   IMU epoch quality check failed - contains NaN")
                            return None
                    else:
                        print(f"   Insufficient IMU data: got {len(accel_data)}, need {expected_imu_samples}")
                        return None
                else:
                    print(f"   No accelerometer data in IMU")
                    return None
            else:
                # Raw array format - assume it's accelerometer data
                if isinstance(imu_data, list):
                    imu_data = np.array(imu_data)
                
                expected_imu_samples = int(4.0 * getattr(self.imu_processor, 'fs', 125))
                
                if len(imu_data) >= expected_imu_samples:
                    imu_epoch = np.zeros((6, expected_imu_samples))
                    raw_epoch = imu_data[-expected_imu_samples:]
                    
                    if raw_epoch.ndim == 1:
                        imu_epoch[0, :] = raw_epoch  # Put in accel_x
                    else:
                        # Try to fit the data appropriately
                        if raw_epoch.shape[0] <= 6:
                            imu_epoch[:raw_epoch.shape[0], :] = raw_epoch
                        else:
                            imu_epoch = raw_epoch[:6, :]  # Take first 6 channels
                    
                    return imu_epoch
                else:
                    print(f"   Insufficient IMU data: got {len(imu_data)}, need {expected_imu_samples}")
                    return None
                    
        except Exception as e:
            print(f"   IMU epoch preparation error: {e}")
            return None

    def _extract_hr_features_for_epoch(self, epoch_start_s, session_duration):
        """FINAL FIXED: Extract HR features with proper time alignment"""
        print(f"❤️  [HR EPOCH] Processing epoch at {epoch_start_s:.1f}s...")
        
        try:
            # Get buffer status first
            hr_status = self.hr_processor.get_buffer_status()
            print(f"   HR Buffer: {hr_status['buffer_size']} samples, time range: {hr_status.get('time_range', 'None')}")
            
            if hr_status['buffer_size'] == 0:
                print(f"   ❌ No HR data in buffer")
                return self._get_default_hr_features()
            
            # FIXED: Convert epoch time to absolute session time
            # epoch_start_s is relative to the processed/trimmed timeline
            # We need to convert it to absolute session time for HR data lookup
            
            # Get absolute session time for this epoch
            absolute_epoch_time = session_duration - 4.0  # Current epoch is 4 seconds before current time
            
            print(f"   Time conversion: epoch_relative={epoch_start_s:.1f}s → session_absolute={absolute_epoch_time:.1f}s")
            
            # Get HR window using absolute time
            epoch_duration = 4.0  # Standard epoch duration
            hr_window = self.hr_processor.get_hr_window_for_epoch(absolute_epoch_time, epoch_duration)
            
            print(f"   HR Window: {len(hr_window)} samples for absolute time {absolute_epoch_time:.1f}-{absolute_epoch_time + epoch_duration:.1f}s")
            
            if len(hr_window) == 0:
                print(f"   ⚠️  No HR samples in epoch window")
                return self._get_default_hr_features()
            
            # FIXED: Better filtering of invalid HR values
            # Filter out invalid HR values (< 30 or > 200 bpm are likely invalid)
            valid_hr_mask = (hr_window >= 30) & (hr_window <= 200)
            valid_hr_data = hr_window[valid_hr_mask]
            
            print(f"   HR Quality: {len(valid_hr_data)}/{len(hr_window)} valid samples (30-200 bpm range)")
            
            if len(valid_hr_data) == 0:
                print(f"   ⚠️  All HR samples are invalid (outside 30-200 bpm range)")
                return self._get_default_hr_features()
            
            # Extract features from valid HR data
            hr_features = self.hr_feature_extractor.extract_features_from_hr_data(valid_hr_data)
            
            # Log feature values
            valid_features = [(k, v) for k, v in hr_features.items() if not pd.isna(v)]
            print(f"   ✅ HR Features: {len(valid_features)}/5 valid")
            if valid_features:
                print(f"      Examples: {valid_features[:3]}")
            
            return hr_features
            
        except Exception as e:
            print(f"   ❌ HR feature extraction failed: {e}")
            import traceback
            traceback.print_exc()
            return self._get_default_hr_features()
    
    def _get_default_hr_features(self):
        """Return default NaN HR features when no data is available"""
        return {
            'heart_rate_87': np.nan,
            'heart_rate_88': np.nan,
            'hr_min': np.nan,
            'hr_max': np.nan,
            'hr_std': np.nan
        }

    def _extract_all_sensor_features(self, processed_eeg_epoch, hr_data, imu_data, epoch_start_s, session_duration):
        """Extract features from ALL sensors (EEG + HR + IMU) to get complete 87-feature set"""
        all_features = {}
        extraction_summary = {'eeg': 0, 'hr': 0, 'imu': 0}
        
        # 1. EEG Features (~56 features)
        print(f"🧠 Extracting EEG features...")
        try:
            eeg_features = self.eeg_feature_extractor.extract_features_from_epoch(processed_eeg_epoch)
            if eeg_features:
                all_features.update(eeg_features)
                extraction_summary['eeg'] = len(eeg_features)
                print(f"   ✅ EEG: {len(eeg_features)} features extracted")
            else:
                print(f"   ❌ EEG: No features returned")
        except Exception as e:
            print(f"   ❌ EEG extraction failed: {e}")
        
        # 2. HR Features - FINAL FIXED VERSION
        print(f"❤️  Extracting HR features...")
        try:
            # FIXED: Use the corrected HR feature extraction method with proper time alignment
            hr_features = self._extract_hr_features_for_epoch(epoch_start_s, session_duration)
            
            if hr_features:
                all_features.update(hr_features)
                extraction_summary['hr'] = len(hr_features)
                
                # Check if any HR features are valid (not NaN)
                valid_hr_features = {k: v for k, v in hr_features.items() if not pd.isna(v)}
                
                if valid_hr_features:
                    print(f"   ✅ HR: {len(hr_features)} features extracted ({len(valid_hr_features)} valid)")
                    # Show sample HR values
                    hr_sample_values = [(k, v) for k, v in valid_hr_features.items()][:3]
                    print(f"   HR values: {hr_sample_values}")
                else:
                    print(f"   ⚠️  HR: {len(hr_features)} features extracted (all NaN - no valid HR data in epoch)")
            else:
                print(f"   ❌ HR: No features returned from processor")
                hr_features = self._get_default_hr_features()
                all_features.update(hr_features)
                extraction_summary['hr'] = len(hr_features)
                
        except Exception as e:
            print(f"   ❌ HR extraction failed: {e}")
            import traceback
            traceback.print_exc()
            # Add default NaN HR features
            hr_features = self._get_default_hr_features()
            all_features.update(hr_features)
            extraction_summary['hr'] = len(hr_features)
        
        # 3. IMU Features (~6 features: stillness_score, movement_intensity, etc.)
        print(f"🏃 Extracting IMU features...")
        try:
            if imu_data is not None and len(imu_data) > 0:
                imu_epoch = self._prepare_imu_epoch(imu_data, epoch_start_s)
                
                if imu_epoch is not None:
                    imu_features = self.imu_feature_extractor.extract_features_from_epoch(imu_epoch)
                    if imu_features:
                        all_features.update(imu_features)
                        extraction_summary['imu'] = len(imu_features)
                        print(f"   ✅ IMU: {len(imu_features)} features extracted")
                        
                        imu_feature_names = list(imu_features.keys())[:3]
                        print(f"   IMU features: {imu_feature_names}...")
                    else:
                        print(f"   ❌ IMU: No features returned from extractor")
                else:
                    print(f"   ⚠️  IMU: Epoch preparation failed")
            else:
                print(f"   ⚠️  IMU: No data provided - will use defaults")
        except Exception as e:
            print(f"   ❌ IMU extraction failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Summary
        total_extracted = sum(extraction_summary.values())
        print(f"📊 MULTI-MODAL EXTRACTION SUMMARY:")
        print(f"   EEG: {extraction_summary['eeg']} features")
        print(f"   HR: {extraction_summary['hr']} features (REPLACES PPG)") 
        print(f"   IMU: {extraction_summary['imu']} features")
        print(f"   TOTAL: {total_extracted} features")
        print(f"   TARGET: 87 features (67 base + 20 smoothed)")
        print(f"   MISSING: {87 - total_extracted} base features (smoothed features will be added during normalization)")
        
        if total_extracted >= 60:
            print(f"   ✅ Good feature extraction coverage!")
        elif total_extracted >= 50:
            print(f"   ⚠️  Partial feature extraction - may impact performance")
        else:
            print(f"   ❌ Poor feature extraction - check sensor data!")
        
        return all_features
    
    def process_realtime_data(self, eeg_data, hr_data=None, imu_data=None, sqc_scores=None, 
                            session_duration=0, streamer_scores=None):
        """Complete multi-modal processing with EEG + HR + IMU features
        
        FINAL FIXED: HR time alignment + normalization loading + better HR filtering
        """
        if eeg_data is None or eeg_data.size == 0:
            return []
        
        print(f"\n{'='*60}")
        print(f"Processing epoch at {session_duration:.1f}s")
        print(f"{'='*60}")
        
        # FIXED: Track session timing for HR alignment
        if self.session_start_absolute_time is None:
            self.session_start_absolute_time = time.time()
        self.last_session_duration = session_duration
        
        # FIXED: Update HR buffer with proper debugging and filtering
        if hr_data is not None:
            print(f"[HR UPDATE] Updating HR buffer with new data...")
            print(f"   Input HR data shape: {hr_data.shape}")
            
            # FIXED: Filter out clearly invalid HR values before adding to buffer
            hr_flat = hr_data.flatten()
            # Count valid vs invalid values
            valid_count = np.sum((hr_flat >= 30) & (hr_flat <= 200))
            invalid_count = np.sum((hr_flat < 30) | (hr_flat > 200))
            print(f"   HR quality: {valid_count} valid, {invalid_count} invalid (30-200 bpm range)")
            
            if valid_count > 0:
                print(f"   Valid HR sample: {hr_flat[(hr_flat >= 30) & (hr_flat <= 200)][:3]}")
            
            self.hr_processor.update_hr_buffer(hr_data, session_duration)
            
            # Show HR buffer status after update
            hr_status = self.hr_processor.get_buffer_status()
            print(f"   HR Buffer: {hr_status['buffer_size']} samples")
            if hr_status['time_range']:
                print(f"   HR Time Range: {hr_status['time_range']}")
            if hr_status['latest_hr']:
                print(f"   Latest HR: {hr_status['latest_hr']:.1f} bpm")
        else:
            print(f"[HR UPDATE] No HR data provided in this call")
        
        # Fix SQC scores shape display
        sqc_shape_str = None
        if sqc_scores is not None:
            if hasattr(sqc_scores, 'shape'):
                sqc_shape_str = str(sqc_scores.shape)
            elif isinstance(sqc_scores, list):
                sqc_shape_str = f"list of {len(sqc_scores)}"
            else:
                sqc_shape_str = str(type(sqc_scores))
        
        # Validate input data shape
        print(f"[INPUT] Data shapes:")
        print(f"  EEG: {eeg_data.shape}")
        print(f"  HR: {hr_data.shape if hr_data is not None and hasattr(hr_data, 'shape') else 'Buffer-based'}")
        print(f"  IMU: {imu_data.shape if imu_data is not None and hasattr(imu_data, 'shape') else 'None'}")
        print(f"  SQC: {sqc_shape_str if sqc_shape_str else 'None'}")
        
        # Handle different EEG data orientations
        print(f"[ORIENTATION] Checking EEG data format...")
        
        # Case 1: Raw EEG from streamer is (samples, 6 channels)
        if eeg_data.ndim == 2 and eeg_data.shape[1] == self.eeg_processor.total_device_channels:
            print(f"  Detected raw EEG format: (samples={eeg_data.shape[0]}, channels={eeg_data.shape[1]})")
            # Transpose to (channels, samples)
            eeg_data = eeg_data.T
            print(f"  Transposed to: {eeg_data.shape}")
            
            # Now select active channels
            if eeg_data.shape[0] == self.eeg_processor.total_device_channels:
                active_indices = self.eeg_processor.active_channel_indices
                eeg_data_active = eeg_data[active_indices, :]
                print(f"  Selected active channels {active_indices}: {eeg_data_active.shape}")
            else:
                print(f"[ERROR] Unexpected channel count after transpose: {eeg_data.shape[0]}")
                return []
        
        # Case 2: Already transposed or filtered EEG (channels, samples)
        elif eeg_data.ndim == 2 and eeg_data.shape[0] == self.eeg_processor.num_active_channels:
            print(f"  Data already in correct format: {eeg_data.shape}")
            eeg_data_active = eeg_data
        
        # Case 3: Filtered EEG with all channels
        elif eeg_data.ndim == 2 and eeg_data.shape[0] == self.eeg_processor.total_device_channels:
            print(f"  Detected full channel data: {eeg_data.shape}")
            active_indices = self.eeg_processor.active_channel_indices
            eeg_data_active = eeg_data[active_indices, :]
            print(f"  Selected active channels {active_indices}: {eeg_data_active.shape}")
        
        # Case 4: Something unexpected
        else:
            print(f"[ERROR] Unexpected EEG shape: {eeg_data.shape}")
            print(f"  Expected either:")
            print(f"    - Raw: (samples, {self.eeg_processor.total_device_channels})")
            print(f"    - Processed: ({self.eeg_processor.num_active_channels}, samples)")
            return []
        
        print(f"[PROCESSING] Using EEG data: {eeg_data_active.shape}")
        # Update session buffer (maintains preprocessing context)
        self._update_session_buffer(eeg_data_active, sqc_scores)
        
        # Check session buffer readiness
        min_buffer_samples = self.eeg_processor.window_samples * 2  # Need at least 2 windows
        if (self.eeg_session_buffer is None or 
            self.eeg_session_buffer.shape[1] < min_buffer_samples):
            buffer_duration = self.eeg_session_buffer.shape[1] / self.eeg_processor.fs if self.eeg_session_buffer is not None else 0
            print(f"[BUFFER] Building context: {buffer_duration:.1f}s / {min_buffer_samples/self.eeg_processor.fs:.1f}s needed")
            return []
        
        print(f"[BUFFER] Ready: {self.eeg_session_buffer.shape[1]} samples ({self.eeg_session_buffer.shape[1]/self.eeg_processor.fs:.1f}s)")
        
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
                
                print(f"\n[EPOCH {epoch_count}] Quality: {'PASS' if final_quality else 'FAIL'} | "
                    f"Type: {'Special' if special_processed else 'Standard'} | "
                    f"Reason: {quality_info['decision_reason']}")
                
                # Extract features for quality-passed epochs
                if final_quality and processed_eeg_epoch is not None:
                    
                    # Validate epoch shape before feature extraction
                    if processed_eeg_epoch.shape != (self.eeg_feature_extractor.num_active_channels, 
                                                    self.eeg_feature_extractor.window_samples):
                        print(f"[WARNING] Epoch shape mismatch: {processed_eeg_epoch.shape}")
                        continue
                    
                    # Extract multi-modal features (EEG + HR + IMU) with FINAL FIXED HR processing
                    print(f"\n[FEATURES] Extracting multi-modal features...")
                    all_sensor_features = self._extract_all_sensor_features(
                        processed_eeg_epoch, hr_data, imu_data, epoch_start_s, session_duration
                    )
                    
                    # Count extracted features by type
                    eeg_count = sum(1 for k in all_sensor_features.keys() 
                                if any(ch in k for ch in ['lf_', 'rf_', 'otel_', 'oter_']))
                    hr_count = sum(1 for k in all_sensor_features.keys() 
                                if any(kw in k for kw in ['heart_rate_87', 'heart_rate_88', 'hr_']))
                    imu_count = sum(1 for k in all_sensor_features.keys() 
                                if any(kw in k for kw in ['stillness', 'movement', 'accel']))
                    
                    print(f"[FEATURES] Extracted: EEG={eeg_count}, HR={hr_count}, IMU={imu_count}")
                    
                    # Apply normalization - FINAL FIXED VERSION
                    print(f"[NORMALIZATION] Applying fixed pre-trained normalization...")
                    normalized_features = self._apply_fixed_normalization(all_sensor_features)
                    
                    # Count smoothed features created
                    smoothed_count = sum(1 for k in normalized_features.keys() if k.endswith('_smoothed'))
                    print(f"[SMOOTHING] Created {smoothed_count} smoothed features")
                    
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
                    
                    # Quick validation
                    feature_count = len([k for k in epoch_features.keys() 
                                    if not k.startswith('epoch_') and not k.endswith('_flag')])
                    nan_count = sum(1 for v in epoch_features.values() 
                                if isinstance(v, (int, float)) and pd.isna(v))
                    
                    print(f"[VALIDATION] Features: {feature_count} | NaN values: {nan_count} | Smoothed: {smoothed_count}")
                    
                    extracted_features.append(epoch_features)
                    
                # Limit epochs per call for real-time processing
                if epoch_count >= 3:
                    print(f"[LIMIT] Processed {epoch_count} epochs, returning for real-time response")
                    break
                        
        except Exception as e:
            print(f"[ERROR] Processing failed: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"\n[COMPLETE] Extracted {len(extracted_features)} quality epochs")
        print(f"  Total processed: {self.processing_stats['total_epochs']}")
        print(f"  Quality passed: {self.processing_stats['software_quality_pass']}")
        print(f"  Normalization: {'Applied' if self.normalization_loaded else 'Not available'}")
        
        return extracted_features
    
    def get_processing_stats(self):
        """Get current processing statistics"""
        return self.processing_stats.copy()
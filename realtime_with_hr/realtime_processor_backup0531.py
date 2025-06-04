# realtime_processor.py - COMPLETE WITH MULTI-MODAL FEATURE EXTRACTION

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
        print(f"   • PPG: Available for heart rate features")
        print(f"   • IMU: Available for movement features")
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
        """Apply fixed pre-trained normalization to features"""
        if not self.normalization_loaded or not self.normalization_stats:
            print("[PROC DEBUG] Normalization skipped: Stats not loaded or empty.")
            return {k.lower(): v for k, v in features_dict_from_extractor.items()}

        processed_features_output = {k.lower(): v for k, v in features_dict_from_extractor.items()}
        
        current_norm_applied_count = 0
        current_smoothed_created_count = 0
        
        power_feature_keywords = ['_abs_', '_rel_', '_power', '_ratio']
        eeg_channel_prefixes = ['lf_', 'rf_', 'otel_', 'oter_'] 
        normalization_log_examples = []

        for feature_name_lower, value in processed_features_output.items():
            if not isinstance(value, (int, float, np.number)) or pd.isna(value):
                continue 

            is_eeg_power_feature = any(feature_name_lower.startswith(prefix) for prefix in eeg_channel_prefixes) and \
                                   any(keyword in feature_name_lower for keyword in power_feature_keywords)

            if is_eeg_power_feature:
                if feature_name_lower in self.normalization_stats:
                    stats = self.normalization_stats[feature_name_lower]
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
                                f"'{feature_name_lower}': {original_value_for_log:.3f} -> {normalized_value:.3f}"
                            )

                        if any(band_keyword in feature_name_lower for band_keyword in ['theta', 'alpha']):
                            smoothed_feature_name_key = f"{feature_name_lower}_smoothed"
                            smoothed_value = self._apply_temporal_smoothing(smoothed_feature_name_key, normalized_value)
                            processed_features_output[smoothed_feature_name_key] = smoothed_value
                            current_smoothed_created_count += 1
                    else:
                        processed_features_output[feature_name_lower] = float(value)
                else:
                    processed_features_output[feature_name_lower] = float(value)
            else:
                if isinstance(value, (int, np.integer)):
                     processed_features_output[feature_name_lower] = float(value)

        if normalization_log_examples:
            print(f"[PROC DETAIL] Normalization applied for {current_norm_applied_count} EEG power features. Examples: {normalization_log_examples}")
        elif current_norm_applied_count > 0:
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
    
    def _prepare_ppg_epoch(self, ppg_data, epoch_start_s):
        """Prepare PPG data epoch for feature extraction"""
        try:
            if isinstance(ppg_data, list):
                ppg_data = np.array(ppg_data)
            
            # PPG epoch should match EEG epoch timing (4 seconds)  
            expected_ppg_samples = int(4.0 * getattr(self.ppg_processor, 'fs', 125))
            
            if len(ppg_data) >= expected_ppg_samples:
                ppg_epoch = ppg_data[-expected_ppg_samples:]
                
                if not np.any(np.isnan(ppg_epoch)) and np.std(ppg_epoch) > 0:
                    return ppg_epoch
                else:
                    print(f"   PPG epoch quality check failed: NaN={np.any(np.isnan(ppg_epoch))}, std={np.std(ppg_epoch):.3f}")
                    return None
            else:
                print(f"   Insufficient PPG data: got {len(ppg_data)}, need {expected_ppg_samples}")
                return None
                
        except Exception as e:
            print(f"   PPG epoch preparation error: {e}")
            return None

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

    def _extract_all_sensor_features(self, processed_eeg_epoch, ppg_data, imu_data, epoch_start_s, session_duration):
        """Extract features from ALL sensors (EEG + PPG + IMU) to get complete 87-feature set"""
        all_features = {}
        extraction_summary = {'eeg': 0, 'ppg': 0, 'imu': 0}
        
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
        
        # 2. PPG Features (~5 features: heart_rate, sdnn, rmssd, etc.)
        print(f"❤️  Extracting PPG features...")
        try:
            if ppg_data is not None and len(ppg_data) > 0:
                ppg_epoch = self._prepare_ppg_epoch(ppg_data, epoch_start_s)
                
                if ppg_epoch is not None:
                    ppg_features = self.ppg_feature_extractor.extract_features_from_epoch(ppg_epoch)
                    if ppg_features:
                        all_features.update(ppg_features)
                        extraction_summary['ppg'] = len(ppg_features)
                        print(f"   ✅ PPG: {len(ppg_features)} features extracted")
                        
                        ppg_feature_names = list(ppg_features.keys())[:3]
                        print(f"   PPG features: {ppg_feature_names}...")
                    else:
                        print(f"   ❌ PPG: No features returned from extractor")
                else:
                    print(f"   ⚠️  PPG: Epoch preparation failed")
            else:
                print(f"   ⚠️  PPG: No data provided - will use defaults")
        except Exception as e:
            print(f"   ❌ PPG extraction failed: {e}")
            import traceback
            traceback.print_exc()
        
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
        print(f"   PPG: {extraction_summary['ppg']} features") 
        print(f"   IMU: {extraction_summary['imu']} features")
        print(f"   TOTAL: {total_extracted} features")
        print(f"   TARGET: 87 features")
        print(f"   MISSING: {87 - total_extracted} (will be filled by FeatureAligner)")
        
        if total_extracted >= 60:
            print(f"   ✅ Good feature extraction coverage!")
        elif total_extracted >= 50:
            print(f"   ⚠️  Partial feature extraction - may impact performance")
        else:
            print(f"   ❌ Poor feature extraction - check sensor data!")
        
        return all_features
    
    def _debug_eeg_feature_extraction(self, processed_eeg_epoch, eeg_features, epoch_count):
        """Comprehensive debug logging for EEG feature extraction"""
        print(f"\n🔍 === EEG FEATURE EXTRACTION DEBUG (Epoch {epoch_count}) ===")
        
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
        
        print(f"\n[DEBUG-2] SHAPE VALIDATION:")
        if actual_shape != expected_shape:
            print(f"   ❌ SHAPE MISMATCH! This causes all NaN features!")
            print(f"   Expected: {expected_shape}")
            print(f"   Got: {actual_shape}")
            print(f"   Difference: channels={actual_shape[0]-expected_shape[0]}, samples={actual_shape[1]-expected_shape[1]}")
        else:
            print(f"   ✅ Shape validation passed")
        
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
                valid_samples = [(k, v) for k, v in eeg_features.items() if not pd.isna(v)][:3]
                for name, value in valid_samples:
                    print(f"      {name}: {value:.6f}")
            else:
                print(f"   ❌ ALL FEATURES ARE NaN!")
                sample_failed = list(eeg_features.keys())[:5]
                print(f"   Sample failed features: {sample_failed}")
        else:
            print(f"   ❌ No features returned!")
        
        print(f"🔍 === END EEG FEATURE EXTRACTION DEBUG ===\n")
    
    def process_realtime_data(self, eeg_data, ppg_data=None, imu_data=None, sqc_scores=None, 
                            session_duration=0, streamer_scores=None):
        """Complete multi-modal processing with EEG + PPG + IMU features"""
        if eeg_data is None or eeg_data.size == 0:
            return []
        
        print(f" Processing with graceful SQC + fixed normalization + MULTI-MODAL features")
        
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
                    
                    print(f"\n🔍 === MULTI-MODAL FEATURE EXTRACTION (Epoch {epoch_count}) ===")
                    
                    # ========== EXTRACT ALL SENSOR FEATURES ==========
                    all_sensor_features = self._extract_all_sensor_features(
                        processed_eeg_epoch, ppg_data, imu_data, epoch_start_s, session_duration
                    )
                    
                    # ========== APPLY NORMALIZATION ==========
                    normalized_features = self._apply_fixed_normalization(all_sensor_features)
                    
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
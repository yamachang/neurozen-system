# realtime_processor.py

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
            key_features = ['LF_rel_theta_power', 'RF_rel_alpha_power', 'LF_rel_alpha_power']
            available_key_features = [f for f in key_features if f in self.normalization_stats]
            print(f"   • Key features available: {len(available_key_features)}/{len(key_features)}")
            
            self.normalization_loaded = True
            return True
            
        except Exception as e:
            print(f"❌ Error loading normalization stats: {e}")
            print(f"   Continuing without normalization")
            self.normalization_loaded = False
            return False
    
    def _apply_fixed_normalization(self, features_dict):
        if not self.normalization_loaded or not self.normalization_stats:
            return features_dict
        
        normalized_features = {}
        normalization_applied_count = 0
        
        # Define which features should be normalized (power features)
        power_feature_keywords = ['_abs_', '_rel_', '_power', '_ratio']
        
        for feature_name, value in features_dict.items():
            # Check if this feature should be normalized
            is_power_feature = any(keyword in feature_name for keyword in power_feature_keywords)
            is_eeg_feature = any(ch in feature_name for ch in ['LF', 'RF', 'OTEL', 'OTER'])
            
            if is_power_feature and is_eeg_feature and feature_name in self.normalization_stats:
                try:
                    stats = self.normalization_stats[feature_name]
                    
                    if (stats.get('initialized', False) and 
                        isinstance(stats.get('mean'), (int, float)) and 
                        isinstance(stats.get('std'), (int, float)) and 
                        stats['std'] > 0):
                        
                        # Apply fixed normalization (inference mode)
                        normalized_value = (value - stats['mean']) / stats['std']
                        normalized_features[feature_name] = normalized_value
                        normalization_applied_count += 1
                        
                        # Apply temporal smoothing for key features (theta, alpha)
                        if any(band in feature_name.lower() for band in ['theta', 'alpha']):
                            smoothed_value = self._apply_temporal_smoothing(feature_name, normalized_value)
                            normalized_features[f"{feature_name}_smoothed"] = smoothed_value
                    else:
                        # Stats not properly initialized, use original value
                        normalized_features[feature_name] = value
                        
                except Exception as e:
                    print(f"   ⚠️  Normalization error for {feature_name}: {e}")
                    normalized_features[feature_name] = value
            else:
                # Not a power feature or no normalization available
                normalized_features[feature_name] = value
        
        self.processing_stats['normalization_applied'] = normalization_applied_count
        return normalized_features
    
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
            artifact_threshold = 0.15  # 25% artifact threshold
            
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
                    # Extract EEG features
                    eeg_features = self.eeg_feature_extractor.extract_features_from_epoch(processed_eeg_epoch)
                    
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
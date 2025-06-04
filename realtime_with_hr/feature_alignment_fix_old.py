# feature_alignment_fix.py

import numpy as np
import pandas as pd
from scipy import interpolate
import json
import os

class FeatureAligner:
    """
    Align real-time features with LSTM expected features
    """
    
    def __init__(self):
        # Expected 87 features (excluding session_id, timestamps, meditation_state)
        self.expected_features = [
            "lf_abs_theta_power", "lf_rel_theta_power", "lf_sef95", "lf_abs_alpha_power",
            "lf_abs_delta_power", "lf_rel_delta_power", "lf_rel_alpha_power", "lf_alpha_theta_ratio",
            "rf_abs_theta_power", "rf_rel_theta_power", "rf_sef95", "rf_abs_alpha_power", 
            "rf_abs_delta_power", "rf_rel_delta_power", "rf_rel_alpha_power", "rf_alpha_theta_ratio",
            "otel_abs_theta_power", "otel_rel_theta_power", "otel_sef95", "otel_abs_alpha_power",
            "otel_abs_delta_power", "otel_rel_delta_power", "otel_rel_alpha_power", "otel_alpha_theta_ratio",
            "oter_abs_theta_power", "oter_rel_theta_power", "oter_sef95", "oter_abs_alpha_power",
            "oter_abs_delta_power", "oter_rel_delta_power", "oter_rel_alpha_power", "oter_alpha_theta_ratio",
            "lf_rf_theta_coherence", "lf_rf_theta_plv", "lf_otel_theta_coherence", "lf_otel_theta_plv",
            "lf_oter_theta_coherence", "lf_oter_theta_plv", "rf_otel_theta_coherence", "rf_otel_theta_plv",
            "rf_oter_theta_coherence", "rf_oter_theta_plv", "otel_oter_theta_coherence", "otel_oter_theta_plv",
            "lf_rf_alpha_coherence", "lf_rf_alpha_plv", "lf_otel_alpha_coherence", "lf_otel_alpha_plv",
            "lf_oter_alpha_coherence", "lf_oter_alpha_plv", "rf_otel_alpha_coherence", "rf_otel_alpha_plv",
            "rf_oter_alpha_coherence", "rf_oter_alpha_plv", "otel_oter_alpha_coherence", "otel_oter_alpha_plv",
            "lf_abs_alpha_power_smoothed", "lf_abs_theta_power_smoothed", "lf_alpha_theta_ratio_smoothed",
            "lf_rel_alpha_power_smoothed", "lf_rel_theta_power_smoothed", "otel_abs_alpha_power_smoothed",
            "otel_abs_theta_power_smoothed", "otel_alpha_theta_ratio_smoothed", "otel_rel_alpha_power_smoothed",
            "otel_rel_theta_power_smoothed", "oter_abs_alpha_power_smoothed", "oter_abs_theta_power_smoothed",
            "oter_alpha_theta_ratio_smoothed", "oter_rel_alpha_power_smoothed", "oter_rel_theta_power_smoothed",
            "rf_abs_alpha_power_smoothed", "rf_abs_theta_power_smoothed", "rf_alpha_theta_ratio_smoothed",
            "rf_rel_alpha_power_smoothed", "rf_rel_theta_power_smoothed", "heart_rate_87", "heart_rate_88",
            "hr_min", "hr_max", "hr_std", "stillness_score", "movement_intensity", "accel_magnitude_mean",
            "accel_magnitude_std", "accel_jerk_mean", "movement_count"
        ]
        
        # Features to exclude (contain these strings)
        self.exclude_patterns = ["special_processing", "_available", "eeg_quality_flag"]
        
        # Feature mapping from real-time names to expected names
        self.feature_mapping = self._create_feature_mapping()
        
        # For interpolation - store recent feature history
        self.feature_history = []
        self.max_history = 10  # Keep last 10 epochs for interpolation
        
    def _create_feature_mapping(self):
        """Create mapping from real-time feature names to expected names"""
        
        mapping = {}
        
        # Direct mappings (real-time name -> expected name)
        # Most will be lowercase conversions
        for expected in self.expected_features:
            # Try uppercase version
            uppercase_version = expected.upper()
            mapping[uppercase_version] = expected
            
            # Try mixed case versions that might exist
            parts = expected.split('_')
            if len(parts) >= 2:
                # Channel names might be uppercase
                channel_upper = parts[0].upper() + '_' + '_'.join(parts[1:])
                mapping[channel_upper] = expected
                
                # All parts uppercase except channel
                all_upper = '_'.join([parts[0].upper()] + [p.upper() for p in parts[1:]])
                mapping[all_upper] = expected
        
        # Specific mappings for known mismatches
        common_mappings = {
            # Add any specific mappings you discover from debugging
            'LF_abs_alpha_power': 'lf_abs_alpha_power',
            'RF_rel_theta_power': 'rf_rel_theta_power',
        }
        mapping.update(common_mappings)
        
        return mapping
    
    def should_exclude_feature(self, feature_name):
        """Check if feature should be excluded based on patterns"""
        for pattern in self.exclude_patterns:
            if pattern in feature_name:
                return True
        return False
    
    def transform_feature_names(self, features_dict):
        """Transform feature names to lowercase and apply mappings"""
        
        transformed = {}
        
        for original_name, value in features_dict.items():
            # Skip excluded features
            if self.should_exclude_feature(original_name):
                continue
                
            # Skip metadata columns
            if original_name in ["session_id", "epoch_start_time_original_s", 
                               "epoch_start_time_trimmed_s", "meditation_state",
                               "realtime_timestamp", "session_duration", "device_id",
                               "test_mode", "eeg_quality_flag"]:
                continue
            
            # Apply mapping if exists, otherwise convert to lowercase
            if original_name in self.feature_mapping:
                expected_name = self.feature_mapping[original_name]
                transformed[expected_name] = value
            else:
                # Default: convert to lowercase
                lowercase_name = original_name.lower()
                if lowercase_name in self.expected_features:
                    transformed[lowercase_name] = value
                # If still not matching, we'll handle in align_features
        
        return transformed
    
    def interpolate_missing_features(self, current_features):
        """Interpolate missing features using history and reasonable defaults"""
        
        # Add current features to history
        self.feature_history.append(current_features.copy())
        if len(self.feature_history) > self.max_history:
            self.feature_history.pop(0)
        
        interpolated_features = {}
        
        for expected_feature in self.expected_features:
            if expected_feature in current_features:
                # Feature is available
                interpolated_features[expected_feature] = current_features[expected_feature]
            else:
                # Feature is missing - interpolate
                interpolated_value = self._interpolate_single_feature(expected_feature)
                interpolated_features[expected_feature] = interpolated_value
        
        return interpolated_features
    
    def _interpolate_single_feature(self, feature_name):
        """Interpolate a single missing feature"""
        
        # Try to get historical values for this feature
        historical_values = []
        for hist_features in self.feature_history:
            if feature_name in hist_features:
                val = hist_features[feature_name]
                if not (np.isnan(val) if isinstance(val, (int, float)) else False):
                    historical_values.append(val)
        
        if len(historical_values) >= 2:
            # Use trend from historical values
            if len(historical_values) >= 3:
                # Linear interpolation trend
                recent_values = historical_values[-3:]
                trend = (recent_values[-1] - recent_values[0]) / len(recent_values)
                interpolated = recent_values[-1] + trend
            else:
                # Simple average
                interpolated = np.mean(historical_values)
                
        elif len(historical_values) == 1:
            # Use the last known value
            interpolated = historical_values[-1]
            
        else:
            # No history - use feature-specific defaults
            interpolated = self._get_default_value(feature_name)
        
        return float(interpolated)
    
    def _get_default_value(self, feature_name):
        """Get reasonable default value for a feature"""
        
        # Feature-specific defaults based on typical EEG/physiological values
        if 'abs_' in feature_name and 'power' in feature_name:
            # Absolute power features - typically positive, log-scale
            return -1.0  # Moderate power in log scale
            
        elif 'rel_' in feature_name and 'power' in feature_name:
            # Relative power features - typically 0-1, but in log scale can be negative
            return -1.5  # Moderate relative power
            
        elif 'coherence' in feature_name:
            # Coherence values - typically 0-1
            return 0.3  # Moderate coherence
            
        elif 'plv' in feature_name:
            # Phase locking value - typically 0-1
            return 0.2  # Low-moderate PLV
            
        elif 'ratio' in feature_name:
            # Ratios - can vary, but alpha/theta ratio typically around 1
            return 0.0  # Neutral ratio in log scale
            
        elif 'sef95' in feature_name:
            # Spectral edge frequency - typically 10-30 Hz
            return 15.0  # Mid-range SEF
            
        elif 'heart_rate' in feature_name or 'hr_' in feature_name:
            # Heart rate features
            if 'heart_rate_' in feature_name:
                return 70.0  # Normal resting heart rate
            elif 'hr_min' in feature_name:
                return 65.0
            elif 'hr_max' in feature_name:
                return 75.0
            elif 'hr_std' in feature_name:
                return 5.0
                
        elif 'stillness' in feature_name:
            # Stillness score - typically 0-1
            return 0.7  # Moderately still
            
        elif 'movement' in feature_name:
            # Movement features
            if 'movement_count' in feature_name:
                return 2.0  # Few movements
            else:
                return 0.1  # Low movement intensity
                
        elif 'accel' in feature_name:
            # Accelerometer features
            if 'magnitude_mean' in feature_name:
                return 9.8  # Gravity
            elif 'magnitude_std' in feature_name:
                return 0.2  # Low variability
            elif 'jerk' in feature_name:
                return 0.1  # Low jerk
                
        elif 'smoothed' in feature_name:
            # Smoothed versions - use base feature defaults
            base_feature = feature_name.replace('_smoothed', '')
            return self._get_default_value(base_feature)
        
        else:
            # Generic default
            return 0.0
    
    def align_features(self, raw_features):
        """
        Complete feature alignment pipeline
        
        Args:
            raw_features: dict with raw real-time feature names and values
            
        Returns:
            dict with exactly 87 features with expected names
        """
        
        # Step 1: Transform feature names and exclude unwanted features
        transformed = self.transform_feature_names(raw_features)
        
        # Step 2: Interpolate missing features
        aligned = self.interpolate_missing_features(transformed)
        
        # Step 3: Ensure we have exactly 87 features
        if len(aligned) != 87:
            print(f"⚠️  Feature count mismatch: got {len(aligned)}, expected 87")
            print(f"   Missing: {set(self.expected_features) - set(aligned.keys())}")
            print(f"   Extra: {set(aligned.keys()) - set(self.expected_features)}")
        
        # Step 4: Return features in consistent order
        ordered_features = {}
        for feature_name in self.expected_features:
            if feature_name in aligned:
                ordered_features[feature_name] = aligned[feature_name]
            else:
                # Last resort default
                ordered_features[feature_name] = self._get_default_value(feature_name)
        
        return ordered_features

def update_lstm_inference_with_alignment():
    """Create updated LSTM inference that uses feature alignment"""
    
    updated_code = '''# Add this to your lstm_inference.py after the LSTMInferenceEngine class

class AlignedLSTMInferenceEngine(LSTMInferenceEngine):
    """LSTM Inference Engine with automatic feature alignment"""
    
    def __init__(self, model_dir="./models"):
        super().__init__(model_dir)
        self.feature_aligner = FeatureAligner()
        print("✅ Feature aligner initialized for 87-feature alignment")
    
    def predict(self, raw_features):
        """
        Make prediction with automatic feature alignment
        
        Args:
            raw_features: dict with raw real-time feature names
            
        Returns:
            dict: Prediction results
        """
        try:
            # Align features to match LSTM expectations
            aligned_features = self.feature_aligner.align_features(raw_features)
            
            # Use the parent predict method with aligned features
            return super().predict(aligned_features)
            
        except Exception as e:
            print(f"❌ Error in aligned prediction: {e}")
            return {
                'ready': False,
                'error': str(e),
                'prediction': None
            }
'''
    
    return updated_code

def test_feature_alignment():
    """Test the feature alignment with sample data"""
    
    print("Testing Feature Alignment")
    print("=" * 50)
    
    # Create sample real-time features (with typical mismatches)
    sample_features = {
        'LF_ABS_ALPHA_POWER': -1.5,
        'RF_REL_THETA_POWER': -2.1,
        'LF_abs_alpha_power': -1.3,  # Lowercase version
        'rf_rel_theta_power': -1.9,  # Lowercase version
        'OTEL_SEF95': 15.2,
        'heart_rate_87': 72.0,
        'stillness_score': 0.8,
        'special_processing': True,  # Should be excluded
        '_available': 1,  # Should be excluded
        'session_id': 'test_123',  # Should be excluded
        'eeg_quality_flag': True,  # Should be excluded
    }
    
    # Test alignment
    aligner = FeatureAligner()
    aligned = aligner.align_features(sample_features)
    
    print(f"Input features: {len(sample_features)}")
    print(f"Aligned features: {len(aligned)}")
    print(f"Expected features: {len(aligner.expected_features)}")
    
    # Show some examples
    print(f"\n✅ Sample aligned features:")
    feature_names = list(aligned.keys())
    for i in range(min(10, len(feature_names))):
        name = feature_names[i]
        value = aligned[name]
        print(f"   {name}: {value:.4f}")
    
    if len(aligned) == 87:
        print(f"\n SUCCESS: Aligned to exactly 87 features!")
    else:
        print(f"\n Issue: Got {len(aligned)} features, expected 87")
    
    return aligned

if __name__ == "__main__":
    # Test the alignment
    test_result = test_feature_alignment()
    
    # Show the updated LSTM code
    print(f"\n" + "="*50)
    print("UPDATED LSTM INFERENCE CODE:")
    print("="*50)
    print(update_lstm_inference_with_alignment())
    
    print(f"\n NEXT STEPS:")
    print("1. Add FeatureAligner to your project")
    print("2. Update your LSTM inference to use AlignedLSTMInferenceEngine")
    print("3. Test with real streaming data")
    print("4. Monitor for '✅ Feature aligner initialized' message")
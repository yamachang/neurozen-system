# feature_alignment_fix.py - EEG + IMU ONLY (NO PPG/HR)

import numpy as np
import pandas as pd
import json
import os

class FeatureAligner:
    def __init__(self, global_stats_path="global_feature_fill_stats.json"):
        # Updated expected features list without HR features (82 total features)
        self.expected_features = [
            # EEG Features (56 features)
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
            # EEG Smoothed Features (20 features)
            "lf_abs_alpha_power_smoothed", "lf_abs_theta_power_smoothed", "lf_alpha_theta_ratio_smoothed",
            "lf_rel_alpha_power_smoothed", "lf_rel_theta_power_smoothed", "otel_abs_alpha_power_smoothed",
            "otel_abs_theta_power_smoothed", "otel_alpha_theta_ratio_smoothed", "otel_rel_alpha_power_smoothed",
            "otel_rel_theta_power_smoothed", "oter_abs_alpha_power_smoothed", "oter_abs_theta_power_smoothed",
            "oter_alpha_theta_ratio_smoothed", "oter_rel_alpha_power_smoothed", "oter_rel_theta_power_smoothed",
            "rf_abs_alpha_power_smoothed", "rf_abs_theta_power_smoothed", "rf_alpha_theta_ratio_smoothed",
            "rf_rel_alpha_power_smoothed", "rf_rel_theta_power_smoothed",
            # IMU Features (6 features) - NO HR FEATURES
            "stillness_score", "movement_intensity", "accel_magnitude_mean",
            "accel_magnitude_std", "accel_jerk_mean", "movement_count"
        ]
        
        # Create a quick lookup for expected features in lowercase
        self.expected_features_lower = {f.lower() for f in self.expected_features}

        print(f"[ALIGNER INFO] Initialized with {len(self.expected_features)} expected features (EEG + IMU only).")
        print(f"[ALIGNER INFO] NO PPG/HR features - streamlined pipeline")

        # Metadata and unwanted patterns to exclude (case-insensitive for patterns)
        self.exclude_keywords_lower = [
            "session_id", "epoch_start_time", "meditation_state", "realtime_timestamp",
            "session_duration", "device_id", "test_mode", "eeg_quality_flag",
            "special_processing", "_available", "quality_assessment_method",
            "quality_decision_reason", "normalization_mode", "normalization_applied",
            "sqc_advisory_info", "streamer_meditation_score", "direct_api_", # Catches all direct_api power bands
            "frenz_audio_status", "_internal_", # Catches internal metadata from realtime_processor
            "heart_rate", "hr_", "ppg_", "bpm"  # Exclude any remaining HR/PPG features
        ]
        
        self.feature_history = []
        self.max_history = 10

        self.global_fill_stats = None
        if global_stats_path and os.path.exists(global_stats_path):
            try:
                with open(global_stats_path, 'r') as f:
                    self.global_fill_stats = json.load(f)
                print(f"[ALIGNER INFO] Successfully loaded global feature fill stats from: {global_stats_path}")
            except Exception as e:
                print(f"[ALIGNER WARNING] Could not load global feature fill stats from {global_stats_path}: {e}")
        else:
            print(f"[ALIGNER INFO] Global feature fill stats file not found: {global_stats_path}. Will use hardcoded defaults if history interpolation fails.")

    def should_exclude_feature(self, feature_name_lower):
        for keyword in self.exclude_keywords_lower:
            if keyword in feature_name_lower:
                return True
        return False

    def transform_feature_names(self, features_dict_from_processor):
        """
        Transforms incoming feature names to the canonical lowercase versions
        expected by the model (EEG + IMU only, no HR).
        """
        transformed = {}
        problematic_names = {} # To log names that couldn't be mapped

        for original_name, value in features_dict_from_processor.items():
            original_name_lower = original_name.lower()

            if self.should_exclude_feature(original_name_lower):
                continue # Skip excluded/metadata features including HR/PPG

            # The canonical names in self.expected_features are already lowercase.
            # So, if original_name_lower is in self.expected_features_lower, we found our match.
            if original_name_lower in self.expected_features_lower:
                # Use the canonical name from self.expected_features that corresponds to this lower_case version
                # This ensures consistent casing if self.expected_features had mixed case (though it shouldn't)
                # For simplicity, assuming self.expected_features are all lowercase as per feature_columns.txt
                transformed[original_name_lower] = value
            else:
                # Log features that couldn't be directly mapped to an expected lowercase name
                if len(problematic_names) < 10 : # Log a few examples
                    problematic_names[original_name] = value

        if problematic_names:
            print(f"[ALIGNER WARNING] transform_feature_names: Could not map some incoming feature names to expected lowercase features. Examples: {problematic_names}")
            print(f"[ALIGNER INFO]   (This may be OK if they are extra features not used by the model, or an issue if expected features are misnamed)")
            
        return transformed

    def _get_default_value(self, feature_name):
        """Get default value for missing features (no HR defaults)"""
        if self.global_fill_stats and feature_name in self.global_fill_stats:
            stats = self.global_fill_stats[feature_name]
            if stats.get('dtype', 'numeric') == 'numeric':
                return float(stats.get('median', stats.get('mean', 0.0)))
            else:
                return stats.get('mode', None)

        # EEG power feature defaults
        if 'abs_' in feature_name and 'power' in feature_name: 
            return -1.0
        if 'rel_' in feature_name and 'power' in feature_name: 
            return -1.5
        if 'alpha_theta_ratio' in feature_name:
            return -0.5
        if 'coherence' in feature_name or 'plv' in feature_name:
            return 0.3
        if 'sef95' in feature_name:
            return 10.0
        
        # IMU feature defaults
        if 'stillness_score' in feature_name: 
            return 0.7
        if 'movement_intensity' in feature_name:
            return 200.0
        if 'accel_magnitude_mean' in feature_name:
            return 150.0
        if 'accel_magnitude_std' in feature_name:
            return 180.0
        if 'accel_jerk_mean' in feature_name:
            return 5000.0
        if 'movement_count' in feature_name: 
            return 2.0
        
        # Smoothed feature defaults
        if '_smoothed' in feature_name:
            base_feature = feature_name.replace('_smoothed', '')
            return self._get_default_value(base_feature)
        
        return 0.0

    def _interpolate_single_feature(self, feature_name):
        """Interpolate single feature from history"""
        historical_values = []
        for hist_features in self.feature_history:
            if feature_name in hist_features and not pd.isna(hist_features[feature_name]):
                historical_values.append(hist_features[feature_name])
        
        if len(historical_values) >= 2:
            return float(np.mean(historical_values[-min(len(historical_values), 3):]))
        elif len(historical_values) == 1:
            return float(historical_values[0])
        else:
            return self._get_default_value(feature_name)

    def interpolate_missing_features(self, current_features_transformed_names):
        """Interpolate missing features (EEG + IMU only, no HR)"""
        interpolated_features = {}
        defaulted_features = []
        interpolated_features_list = []
        
        for expected_feature in self.expected_features: # self.expected_features are canonical lowercase
            if expected_feature in current_features_transformed_names and \
               not pd.isna(current_features_transformed_names[expected_feature]):
                interpolated_features[expected_feature] = current_features_transformed_names[expected_feature]
            else: 
                interpolated_value = self._interpolate_single_feature(expected_feature)
                interpolated_features[expected_feature] = interpolated_value
                
                if len(self.feature_history) == 0:
                    defaulted_features.append(expected_feature)
                else:
                    interpolated_features_list.append(expected_feature)
        
        if defaulted_features:
            print(f"[ALIGNER DETAIL] Used default values for {len(defaulted_features)} features (first epoch)")
            if len(defaulted_features) <= 5:
                print(f"   Examples: {defaulted_features[:5]}")
        
        if interpolated_features_list:
            print(f"[ALIGNER DETAIL] Interpolated {len(interpolated_features_list)} features from history")
            if len(interpolated_features_list) <= 3:
                print(f"   Examples: {interpolated_features_list[:3]}")
        
        self.feature_history.append(interpolated_features.copy())
        if len(self.feature_history) > self.max_history:
            self.feature_history.pop(0)
            
        return interpolated_features

    def align_features(self, raw_features_from_processor):
        """Align features for EEG + IMU only model (no HR)"""
        print(f"[ALIGNER DEBUG] === Aligning New Epoch (EEG + IMU only) ===")
        
        # Remove any HR/PPG features that might have slipped through
        cleaned_features = {}
        hr_ppg_features_found = []
        
        for key, value in raw_features_from_processor.items():
            key_lower = key.lower()
            if any(hr_keyword in key_lower for hr_keyword in ['heart_rate', 'hr_', 'ppg_', 'bpm']):
                hr_ppg_features_found.append(key)
            else:
                cleaned_features[key] = value
        
        if hr_ppg_features_found:
            print(f"[ALIGNER INFO] Removed {len(hr_ppg_features_found)} HR/PPG features: {hr_ppg_features_found[:5]}")
        
        print(f"[ALIGNER DEBUG] Raw features after HR/PPG removal: {len(cleaned_features)}")
        
        transformed_named_features = self.transform_feature_names(cleaned_features)
        print(f"[ALIGNER DEBUG] Transformed features: {len(transformed_named_features)}")
        
        aligned_interpolated_features = self.interpolate_missing_features(transformed_named_features)
        print(f"[ALIGNER DEBUG] Aligned features: {len(aligned_interpolated_features)}")
        
        ordered_features_final = {}
        for feature_name in self.expected_features: # Iterate using canonical lowercase names
            value = aligned_interpolated_features.get(feature_name)
            if value is None: # Should ideally be handled by interpolate_missing_features
                 print(f"[ALIGNER WARNING] Feature '{feature_name}' was None after interpolation. Using final default.")
                 value = self._get_default_value(feature_name)
            ordered_features_final[feature_name] = value
        
        # Final validation
        expected_count = len(self.expected_features)
        actual_count = len(ordered_features_final)
        nan_count = sum(1 for v in ordered_features_final.values() if pd.isna(v))
        
        print(f"[ALIGNER DEBUG] Final validation: {actual_count}/{expected_count} features, {nan_count} NaN values")
        
        if actual_count != expected_count:
            print(f"[ALIGNER ERROR] Feature count mismatch! Expected {expected_count}, got {actual_count}")
        
        if nan_count > 0:
            print(f"[ALIGNER WARNING] {nan_count} features still have NaN values")
            nan_features = [k for k, v in ordered_features_final.items() if pd.isna(v)][:5]
            print(f"   Examples: {nan_features}")
        
        print(f"[ALIGNER SUCCESS] Aligned {actual_count} EEG + IMU features (no HR)")
        
        return ordered_features_final
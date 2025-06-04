# feature_alignment_fix.py

import numpy as np
import pandas as pd
import json
import os

class FeatureAligner:
    def __init__(self, global_stats_path="global_feature_fill_stats.json"):
        self.expected_features = [ # Taken from your feature_columns.txt
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
        # Create a quick lookup for expected features in lowercase
        self.expected_features_lower = {f.lower() for f in self.expected_features}

        print(f"[ALIGNER INFO] Initialized with {len(self.expected_features)} expected features.")

        # Metadata and unwanted patterns to exclude (case-insensitive for patterns)
        self.exclude_keywords_lower = [
            "session_id", "epoch_start_time", "meditation_state", "realtime_timestamp",
            "session_duration", "device_id", "test_mode", "eeg_quality_flag",
            "special_processing", "_available", "quality_assessment_method",
            "quality_decision_reason", "normalization_mode", "normalization_applied",
            "sqc_advisory_info", "streamer_meditation_score", "direct_api_", # Catches all direct_api power bands
            "frenz_audio_status", "_internal_" # Catches internal metadata from realtime_processor
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
        expected by the model.
        """
        transformed = {}
        problematic_names = {} # To log names that couldn't be mapped

        for original_name, value in features_dict_from_processor.items():
            original_name_lower = original_name.lower()

            if self.should_exclude_feature(original_name_lower):
                continue # Skip excluded/metadata features

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
        # (This method can remain as previously revised, using global_fill_stats or hardcoded)
        if self.global_fill_stats and feature_name in self.global_fill_stats:
            stats = self.global_fill_stats[feature_name]
            if stats.get('dtype', 'numeric') == 'numeric':
                return float(stats.get('median', stats.get('mean', 0.0)))
            else:
                return stats.get('mode', None)

        if 'abs_' in feature_name and 'power' in feature_name: return -1.0
        if 'rel_' in feature_name and 'power' in feature_name: return -1.5
        # ... (other hardcoded defaults as before) ...
        if 'heart_rate_87' == feature_name or 'heart_rate_88' == feature_name : return 70.0
        if 'hr_min' in feature_name: return 65.0
        if 'hr_max' in feature_name: return 75.0
        if 'hr_std' in feature_name: return 5.0
        if 'stillness_score' in feature_name: return 0.7
        if 'movement_count' in feature_name: return 2.0
        # ...
        return 0.0

    def _interpolate_single_feature(self, feature_name):
        # (This method can remain as previously revised)
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
        # (This method can remain as previously revised, with its logging)
        interpolated_features = {}
        # ... (logging variables for defaulted/interpolated features) ...

        for expected_feature in self.expected_features: # self.expected_features are canonical lowercase
            if expected_feature in current_features_transformed_names and \
               not pd.isna(current_features_transformed_names[expected_feature]):
                interpolated_features[expected_feature] = current_features_transformed_names[expected_feature]
            else: 
                interpolated_value = self._interpolate_single_feature(expected_feature)
                interpolated_features[expected_feature] = interpolated_value
                # ... (Add your detailed logging here about how the feature was filled) ...
        
        self.feature_history.append(interpolated_features.copy())
        if len(self.feature_history) > self.max_history:
            self.feature_history.pop(0)
            
        return interpolated_features

    def align_features(self, raw_features_from_processor):
        # (This method can remain as previously revised, with its detailed logging structure)
        # It calls transform_feature_names then interpolate_missing_features
        print(f"[ALIGNER DEBUG] === Aligning New Epoch ===")
        # ... (logging as in previous version) ...
        
        transformed_named_features = self.transform_feature_names(raw_features_from_processor)
        # ... (logging) ...
        
        aligned_interpolated_features = self.interpolate_missing_features(transformed_named_features)
        # ... (logging) ...
        
        ordered_features_final = {}
        for feature_name in self.expected_features: # Iterate using canonical lowercase names
            value = aligned_interpolated_features.get(feature_name)
            if value is None: # Should ideally be handled by interpolate_missing_features
                 print(f"[ALIGNER WARNING] Feature '{feature_name}' was None after interpolation. Using final default.")
                 value = self._get_default_value(feature_name)
            ordered_features_final[feature_name] = value
        # ... (final checks and logging) ...
        return ordered_features_final
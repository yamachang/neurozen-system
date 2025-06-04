# src/feature_extraction/hr_features.py
import numpy as np
import pandas as pd

class HRFeatureExtractor:
    def __init__(self):
        """
        HR feature extractor for real-time system
        
        Extracts the exact 5 features needed: heart_rate_87, heart_rate_88, hr_min, hr_max, hr_std
        """
        # Define HR feature names (must match offline processing)
        self.feature_names = [
            'heart_rate_87',  # Primary heart rate measurement
            'heart_rate_88',  # Secondary heart rate measurement (duplicate)
            'hr_min',         # Minimum heart rate in window
            'hr_max',         # Maximum heart rate in window
            'hr_std'          # Heart rate standard deviation
        ]
        
        print(f"HRFeatureExtractor Initialized:")
        print(f"  Features: {self.feature_names}")
    
    def extract_features_from_hr_data(self, hr_window):
        """
        Extract HR features from a window of HR data
        
        Args:
            hr_window (np.array): Array of HR values for the current epoch
        
        Returns:
            dict: HR features matching offline processing
        """
        # Initialize features with NaN (default when no valid data)
        features = {fname: np.nan for fname in self.feature_names}
        
        # Check if we have any HR data
        if hr_window is None or len(hr_window) == 0:
            return features
        
        # Convert to numpy array if needed
        if not isinstance(hr_window, np.ndarray):
            hr_window = np.array(hr_window)
        
        # Filter out invalid values (negative HR) - matches offline logic
        valid_hr = hr_window >= 0
        
        if not np.any(valid_hr):
            return features
        
        filtered_hr = hr_window[valid_hr]
        
        # Calculate features - matches offline processing exactly
        try:
            hr_mean = np.mean(filtered_hr)
            
            features.update({
                'heart_rate_87': hr_mean,  # Primary HR measurement
                'heart_rate_88': hr_mean,  # Secondary HR measurement (duplicate for compatibility)
                'hr_min': np.min(filtered_hr),
                'hr_max': np.max(filtered_hr),
                'hr_std': np.std(filtered_hr) if len(filtered_hr) > 1 else 0.0
            })
            
        except Exception as e:
            print(f"   HR feature extraction error: {e}")
            # Features remain NaN on error
        
        return features
    
    def extract_features_from_epoch(self, hr_window):
        """
        Extract HR features from epoch data (matches interface of other feature extractors)
        
        Args:
            hr_window: HR data for the current epoch
        
        Returns:
            dict: HR features
        """
        return self.extract_features_from_hr_data(hr_window)
    
    def validate_features(self, features):
        """
        Validate extracted HR features
        
        Args:
            features (dict): Extracted HR features
        
        Returns:
            dict: Validation results
        """
        validation = {
            'valid_feature_count': 0,
            'nan_feature_count': 0,
            'feature_status': {}
        }
        
        for feature_name in self.feature_names:
            value = features.get(feature_name, np.nan)
            
            if pd.isna(value):
                validation['nan_feature_count'] += 1
                validation['feature_status'][feature_name] = 'NaN'
            else:
                validation['valid_feature_count'] += 1
                validation['feature_status'][feature_name] = f'{value:.1f}'
        
        validation['success_rate'] = validation['valid_feature_count'] / len(self.feature_names)
        
        return validation
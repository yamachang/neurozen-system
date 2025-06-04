# src/feature_extraction/imu_features.py
import numpy as np
from src.utils.config_manager import load_config
from src.utils.signal_utils import calculate_vector_magnitude, calculate_imu_stillness

class IMUFeatureExtractor:
    def __init__(self):
        config = load_config()
        self.imu_settings = config['signal_processing']['imu']
        
        # Basic parameters
        self.fs = self.imu_settings['sample_rate']
        self.window_sec = self.imu_settings['movement_analysis_window_sec']
        self.stillness_threshold = self.imu_settings['stillness_threshold']
        self.gravity = self.imu_settings['accel_gravity_value']
        
        # Additional parameters from old implementation
        self.epsilon = self.imu_settings.get('stillness_epsilon', 1e-9)
        
        # Define comprehensive IMU feature names
        self.feature_names = [
            'stillness_score',             # Primary metric for meditation depth
            'movement_intensity',          # Overall activity level
            'accel_magnitude_mean',        # Average acceleration magnitude
            'accel_magnitude_std',         # Variability in acceleration
            'accel_jerk_mean',             # Rate of change in acceleration
            # 'posture_stability',           # Based on gyroscope data
            'movement_count'               # Number of discrete movements
        ]
        
        print(f"IMUFeatureExtractor Initialized with {len(self.feature_names)} features")
    
    def extract_features_from_epoch(self, imu_epoch):
        """Extract comprehensive IMU features from a single epoch"""
        features = {fname: np.nan for fname in self.feature_names}
        
        if imu_epoch is None or np.all(np.isnan(imu_epoch)) or imu_epoch.ndim != 2 or imu_epoch.shape[0] < 3:
            return features
        
        try:
            # Extract accelerometer and gyroscope data
            acc_data = imu_epoch[0:3, :]  # First 3 channels are acc_x, acc_y, acc_z
            gyro_data = imu_epoch[3:6, :] if imu_epoch.shape[0] >= 6 else None
            
            # Calculate acceleration magnitude
            try:
                # Use the specialized function if available
                acc_mag = calculate_vector_magnitude(acc_data[0], acc_data[1], acc_data[2])
            except:
                # Fallback calculation
                acc_mag = np.sqrt(np.sum(acc_data**2, axis=0))
            
            # Remove gravity component for acceleration-based features
            acc_mag_norm = acc_mag - self.gravity  # Remove gravity component
            
            # 1. Stillness score - using both approaches for robustness
            try:
                # Use specialized function if available
                features['stillness_score'] = calculate_imu_stillness(acc_mag, epsilon=self.epsilon)
            except:
                # Fallback calculation
                below_threshold = np.abs(acc_mag_norm) < self.stillness_threshold
                features['stillness_score'] = np.mean(below_threshold)
            
            # 2. Movement intensity - standard deviation of acceleration
            features['movement_intensity'] = np.std(acc_mag_norm)
            
            # 3. Acceleration magnitude statistics
            features['accel_magnitude_mean'] = np.mean(acc_mag)
            features['accel_magnitude_std'] = np.std(acc_mag)
            
            # 4. Jerk (rate of change of acceleration)
            acc_diff = np.diff(acc_mag) * self.fs  # Scale by sampling rate to get per-second
            features['accel_jerk_mean'] = np.mean(np.abs(acc_diff))
            
            # 5. Posture stability (from gyroscope if available)
            # if gyro_data is not None:
            #     gyro_mag = np.sqrt(np.sum(gyro_data**2, axis=0))
            #     features['posture_stability'] = 1.0 / (1.0 + np.mean(gyro_mag))
            
            # 6. Movement count (crude approximation of discrete movements)
            # A movement is defined as crossing from below to above the stillness threshold
            movement_transitions = np.diff((np.abs(acc_mag_norm) > self.stillness_threshold).astype(int))
            features['movement_count'] = np.sum(movement_transitions > 0)
            
        except Exception as e:
            print(f"IMU feature extraction error: {e}")
            
        return features
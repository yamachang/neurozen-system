# src/data_processing/imu_processor.py
import numpy as np
from scipy.signal import butter, filtfilt, detrend, savgol_filter
from src.utils.config_manager import load_config

class IMUProcessor:
    def __init__(self):
        config = load_config()
        self.imu_settings = config['signal_processing']['imu']
        
        # Basic parameters
        self.fs = self.imu_settings['sample_rate']
        self.window_sec = self.imu_settings['movement_analysis_window_sec']
        self.overlap_ratio = self.imu_settings['feature_overlap_ratio']
        self.window_samples = int(self.window_sec * self.fs)
        self.step_samples = int(self.window_samples * (1.0 - self.overlap_ratio))
        
        # Stillness and processing parameters
        self.stillness_threshold = self.imu_settings['stillness_threshold']
        self.gravity = self.imu_settings['accel_gravity_value']
        self.epsilon = self.imu_settings.get('stillness_epsilon', 1e-9)
        
        # Filter parameters
        self.lowpass_cutoff = self.imu_settings.get('lowpass_cutoff_hz', 5.0)  # Default 5Hz lowpass for motion
        self.filter_order = self.imu_settings.get('filter_order', 4)  # Default 4th order Butterworth
        
        # Expected channel structure
        self.channel_structure = {
            'acc_x': 0,
            'acc_y': 1, 
            'acc_z': 2,
            'gyro_x': 3,
            'gyro_y': 4,
            'gyro_z': 5
        }
        
        # Initialize debug statistics
        self.movement_stats = {
            'total_samples_processed': 0,
            'detected_movements': 0,
            'quiet_periods': 0,
            'avg_movement_duration_s': 0
        }
        
        print(f"IMUProcessor Initialized.")
        print(f"  IMU Fs: {self.fs} Hz, Lowpass Cutoff: {self.lowpass_cutoff} Hz")
        print(f"  Epoching: {self.window_sec}s window, {self.step_samples/self.fs:.2f}s step")
        print(f"  Stillness threshold: {self.stillness_threshold}, Gravity: {self.gravity}")
    
    def preprocess_imu(self, raw_imu_data):
        """
        Apply comprehensive preprocessing to raw IMU data
        
        Args:
            raw_imu_data (np.ndarray): Raw IMU data, shape (channels, samples) or (samples, channels)
            
        Returns:
            np.ndarray: Preprocessed IMU data, shape (channels, samples)
        """
        # Check data format and shape
        if raw_imu_data is None:
            print(f"IMU data is None")
            return None
            
        # Handle potential shape variations
        if raw_imu_data.ndim != 2:
            print(f"IMU data is not 2D, shape: {raw_imu_data.shape}")
            return None
            
        # Ensure data is in (channels, samples) format
        if raw_imu_data.shape[0] > raw_imu_data.shape[1]:
            # More likely (samples, channels) format, so transpose
            imu_data = raw_imu_data.T
            print(f"Transposed IMU data from {raw_imu_data.shape} to {imu_data.shape}")
        else:
            imu_data = raw_imu_data
            
        # Ensure minimum required channels (at least 3 for accelerometer)
        min_channels = 3
        if imu_data.shape[0] < min_channels:
            print(f"IMU data has insufficient channels: {imu_data.shape[0]}, need at least {min_channels}")
            return None
        
        try:
            # Create output array
            processed_data = np.zeros_like(imu_data)
            num_channels = imu_data.shape[0]
            
            # Determine channels to process
            acc_channels = list(range(3))  # First three channels are acc_x, acc_y, acc_z
            gyro_channels = list(range(3, 6)) if num_channels >= 6 else []
            
            # Process accelerometer channels
            for i in acc_channels:
                # 1. De-trend to remove slow drift
                detrended = detrend(imu_data[i, :], type='linear')
                
                # 2. Apply low-pass filter
                b, a = butter(self.filter_order, self.lowpass_cutoff/(self.fs/2), 'low')
                filtered = filtfilt(b, a, detrended)
                
                # 3. Optional: Smoothing with Savitzky-Golay filter for noise reduction
                if len(filtered) > 15:  # Need enough points
                    try:
                        # Adaptive window size based on data length
                        window_length = min(15, len(filtered) // 10 * 2 + 1)  # Ensure odd
                        if window_length >= 5:  # Minimum valid window
                            smoothed = savgol_filter(filtered, window_length, 2)
                            processed_data[i, :] = smoothed
                        else:
                            processed_data[i, :] = filtered
                    except:
                        processed_data[i, :] = filtered
                else:
                    processed_data[i, :] = filtered
            
            # Process gyroscope channels if present
            for i in gyro_channels:
                # 1. De-trend to remove slow drift
                detrended = detrend(imu_data[i, :], type='linear')
                
                # 2. Apply low-pass filter
                b, a = butter(self.filter_order, self.lowpass_cutoff/(self.fs/2), 'low')
                filtered = filtfilt(b, a, detrended)
                
                # Store in output
                processed_data[i, :] = filtered
            
            # Track statistics for debugging
            self.movement_stats['total_samples_processed'] += imu_data.shape[1]
            
            # Detect movements in the processed data
            acc_mag = np.sqrt(np.sum(processed_data[acc_channels, :]**2, axis=0))
            acc_mag_norm = acc_mag - self.gravity
            movements = np.abs(acc_mag_norm) > self.stillness_threshold
            
            # Count movement transitions (going from still to moving)
            movement_transitions = np.diff(movements.astype(int))
            movement_starts = np.where(movement_transitions > 0)[0]
            movement_ends = np.where(movement_transitions < 0)[0]
            
            # Update movement statistics
            num_movements = len(movement_starts)
            self.movement_stats['detected_movements'] += num_movements
            
            # Calculate average movement duration if possible
            if num_movements > 0 and len(movement_ends) > 0:
                # Match starts with ends
                valid_pairs = 0
                total_duration = 0
                for start in movement_starts:
                    # Find the next end after this start
                    ends_after_start = movement_ends[movement_ends > start]
                    if len(ends_after_start) > 0:
                        duration = (ends_after_start[0] - start) / self.fs  # In seconds
                        total_duration += duration
                        valid_pairs += 1
                
                if valid_pairs > 0:
                    avg_duration = total_duration / valid_pairs
                    # Moving average update of stored stats
                    old_avg = self.movement_stats['avg_movement_duration_s']
                    weight = 0.1  # Low weight to smooth across sessions
                    self.movement_stats['avg_movement_duration_s'] = (
                        (1-weight) * old_avg + weight * avg_duration if old_avg > 0 else avg_duration
                    )
            
            print(f"  Processed IMU data: {num_channels} channels, {imu_data.shape[1]} samples")
            if num_movements > 0:
                print(f"  Detected {num_movements} movements in this segment")
            
            return processed_data
            
        except Exception as e:
            print(f"IMU preprocessing error: {e}")
            # Return original data if processing fails
            return imu_data
    
    def detect_movement_events(self, processed_imu_data):
        """
        Detect movement events in processed IMU data
        
        Args:
            processed_imu_data (np.ndarray): Processed IMU data, shape (channels, samples)
            
        Returns:
            tuple: (movement_mask, event_starts, event_ends)
        """
        if processed_imu_data is None or processed_imu_data.shape[0] < 3:
            return None, None, None
            
        # Calculate acceleration magnitude
        acc_mag = np.sqrt(np.sum(processed_imu_data[0:3, :]**2, axis=0))
        acc_mag_norm = acc_mag - self.gravity
        
        # Movement mask
        movement_mask = np.abs(acc_mag_norm) > self.stillness_threshold
        
        # Detect movement events
        movement_transitions = np.diff(movement_mask.astype(int))
        event_starts = np.where(movement_transitions > 0)[0]
        event_ends = np.where(movement_transitions < 0)[0]
        
        return movement_mask, event_starts, event_ends
    
    def generate_imu_epochs(self, imu_data_trimmed):
        """
        Generate epochs from trimmed IMU data
        
        Args:
            imu_data_trimmed (np.ndarray): Trimmed and preprocessed IMU data
            
        Yields:
            tuple: (imu_window, epoch_start_s, movement_info)
                - imu_window: IMU data for this epoch
                - epoch_start_s: Start time of epoch in seconds
                - movement_info: Dict with movement metrics for this epoch
        """
        if imu_data_trimmed is None or \
           imu_data_trimmed.ndim != 2 or \
           imu_data_trimmed.shape[1] < self.window_samples:
            return
        
        total_samples = imu_data_trimmed.shape[1]
        
        for start_idx in range(0, total_samples - self.window_samples + 1, self.step_samples):
            # Extract window
            current_window = imu_data_trimmed[:, start_idx : start_idx + self.window_samples]
            
            # Calculate epoch metrics
            movement_mask, event_starts, event_ends = self.detect_movement_events(current_window)
            
            # Movement information for this epoch
            movement_info = {
                'movement_ratio': np.mean(movement_mask) if movement_mask is not None else np.nan,
                'num_events': len(event_starts) if event_starts is not None else 0
            }
            
            yield current_window, start_idx / self.fs, movement_info
# src/data_processing/hr_processor.py
import numpy as np
from collections import deque
import time

class HRProcessor:
    def __init__(self):
        """
        HR data processor for real-time system
        
        HR data from Frenz API comes at 0.2 Hz (1 sample per 5 seconds)
        """
        # HR data characteristics from Frenz API
        self.hr_sample_rate = 0.2  # Hz (1 sample per 5 seconds)
        self.sample_interval = 1.0 / self.hr_sample_rate  # 5 seconds between samples
        
        # Processing window
        self.epoch_window_sec = 4.0  # Match EEG processing window
        
        # Real-time buffer management
        self.hr_buffer = deque(maxlen=100)  # Keep last 100 HR samples (~8.3 minutes)
        self.buffer_start_time = None
        
        print(f"HRProcessor Initialized:")
        print(f"  HR Sample Rate: {self.hr_sample_rate} Hz (1 sample per {self.sample_interval} seconds)")
        print(f"  Processing Window: {self.epoch_window_sec} seconds")
        print(f"  Expected samples per epoch: {max(1, int(self.epoch_window_sec * self.hr_sample_rate))}")
    
    def update_hr_buffer(self, hr_array, current_session_time):
        """
        Update HR buffer with new data from Frenz API
        
        Args:
            hr_array: Array of HR values from streamer.SCORES["hr"] or similar
            current_session_time: Current session time in seconds
        """
        if hr_array is None or len(hr_array) == 0:
            return
        
        # Convert to numpy array if needed
        if not isinstance(hr_array, np.ndarray):
            hr_array = np.array(hr_array)
        
        # For real-time processing, each call typically has recent HR samples
        # Calculate timestamps based on current session time working backwards
        for i, hr_value in enumerate(hr_array):
            # Calculate timestamp for this HR sample
            # Most recent sample gets current_session_time, earlier samples get earlier timestamps
            sample_timestamp = current_session_time - (len(hr_array) - 1 - i) * self.sample_interval
            
            self.hr_buffer.append({
                'timestamp': sample_timestamp,
                'hr_value': float(hr_value),
                'session_time': current_session_time
            })
    
    def update_hr_buffer_batch(self, hr_array, session_start_time):
        """
        Update HR buffer with a batch of HR data (for offline-style processing)
        
        Args:
            hr_array: Complete array of HR values from session start
            session_start_time: Session start time in seconds
        """
        if hr_array is None or len(hr_array) == 0:
            return
        
        # Clear existing buffer for batch update
        self.hr_buffer.clear()
        
        # Add all samples with correct timestamps from session start
        for i, hr_value in enumerate(hr_array):
            sample_timestamp = session_start_time + i * self.sample_interval
            
            self.hr_buffer.append({
                'timestamp': sample_timestamp,
                'hr_value': float(hr_value),
                'session_time': session_start_time + len(hr_array) * self.sample_interval
            })
    
    def get_hr_window_for_epoch(self, epoch_start_time, epoch_duration=None):
        """
        Get HR data window for a specific epoch
        
        Args:
            epoch_start_time: Start time of the epoch in seconds
            epoch_duration: Duration of the epoch (default: self.epoch_window_sec)
        
        Returns:
            np.array: HR values within the epoch window
        """
        if epoch_duration is None:
            epoch_duration = self.epoch_window_sec
        
        epoch_end_time = epoch_start_time + epoch_duration
        
        # Find HR samples that fall within this epoch
        hr_values_in_window = []
        
        for hr_sample in self.hr_buffer:
            sample_time = hr_sample['timestamp']
            
            # Check if this sample falls within the epoch window
            if epoch_start_time <= sample_time < epoch_end_time:
                hr_values_in_window.append(hr_sample['hr_value'])
        
        return np.array(hr_values_in_window) if hr_values_in_window else np.array([])
    
    def process_hr_for_epoch(self, epoch_start_time, epoch_duration=None):
        """
        Process HR data for a specific epoch - matches offline processing logic exactly
        
        Args:
            epoch_start_time: Start time of the epoch in seconds
            epoch_duration: Duration of the epoch (default: self.epoch_window_sec)
        
        Returns:
            dict: HR features for this epoch
        """
        if epoch_duration is None:
            epoch_duration = self.epoch_window_sec
        
        # Get HR window for this epoch
        hr_window = self.get_hr_window_for_epoch(epoch_start_time, epoch_duration)
        
        # Default features (all NaN if no valid data)
        hr_features = {
            'heart_rate_87': np.nan,
            'heart_rate_88': np.nan,  # Duplicate for compatibility
            'hr_min': np.nan,
            'hr_max': np.nan,
            'hr_std': np.nan
        }
        
        if len(hr_window) == 0:
            return hr_features
        
        # Filter out invalid values (negative HR) - matches offline logic
        valid_hr = hr_window >= 0
        
        if not np.any(valid_hr):
            return hr_features
        
        filtered_hr = hr_window[valid_hr]
        
        # Calculate features - matches offline logic exactly
        hr_mean = np.mean(filtered_hr)
        hr_features.update({
            'heart_rate_87': hr_mean,  # Primary HR measurement
            'heart_rate_88': hr_mean,  # Secondary HR measurement (duplicate)
            'hr_min': np.min(filtered_hr),
            'hr_max': np.max(filtered_hr),
            'hr_std': np.std(filtered_hr) if len(filtered_hr) > 1 else 0.0
        })
        
        return hr_features
    
    def get_buffer_status(self):
        """Get current buffer status for debugging"""
        if not self.hr_buffer:
            return {
                'buffer_size': 0,
                'time_range': None,
                'latest_hr': None
            }
        
        timestamps = [sample['timestamp'] for sample in self.hr_buffer]
        hr_values = [sample['hr_value'] for sample in self.hr_buffer]
        
        return {
            'buffer_size': len(self.hr_buffer),
            'time_range': f"{min(timestamps):.1f} - {max(timestamps):.1f}s",
            'latest_hr': hr_values[-1] if hr_values else None,
            'hr_range': f"{min(hr_values):.1f} - {max(hr_values):.1f} bpm" if hr_values else None
        }
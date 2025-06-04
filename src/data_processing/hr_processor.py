# src/data_processing/hr_processor.py - FINAL FIX: TIME ALIGNMENT RESOLVED

import numpy as np
import pandas as pd
from collections import deque
import time

class HRProcessor:
    def __init__(self, buffer_duration_minutes=30):
        """Initialize HR processor with buffering for real-time processing"""
        
        # HR data specifications (from Frenz API)
        self.hr_sample_rate = 0.2  # 1 sample every 5 seconds
        self.hr_sample_interval = 5.0  # seconds between samples
        
        # Real-time buffer management
        self.buffer_duration_s = buffer_duration_minutes * 60
        self.max_buffer_samples = int(self.buffer_duration_s / self.hr_sample_interval)
        
        # Initialize buffers - using deque for efficient FIFO operations
        self.hr_values = deque(maxlen=self.max_buffer_samples)  # HR values (bpm)
        self.hr_timestamps = deque(maxlen=self.max_buffer_samples)  # Corresponding timestamps
        
        # FIXED: Track session timing for proper alignment
        self.session_start_time = None
        self.total_samples_received = 0
        
        print(f"✅ HR Processor initialized:")
        print(f"   • Sample rate: {self.hr_sample_rate} Hz (1 sample per {self.hr_sample_interval}s)")
        print(f"   • Buffer duration: {buffer_duration_minutes} minutes")
        print(f"   • Max buffer samples: {self.max_buffer_samples}")
        print(f"   • Processing mode: Real-time incremental with time alignment")

    def update_hr_buffer(self, new_hr_data, current_session_time):
        """Update HR buffer with new data from real-time streaming - FINAL FIXED VERSION
        
        Args:
            new_hr_data: numpy array of HR values from streamer (cumulative array)
            current_session_time: current session time in seconds
        """
        if new_hr_data is None or len(new_hr_data) == 0:
            return
        
        # Handle different input formats
        if isinstance(new_hr_data, list):
            new_hr_data = np.array(new_hr_data)
        
        # Flatten if needed (handle shape like (27, 1))
        new_hr_data = new_hr_data.flatten()
        
        # FIXED: Initialize session start time on first data
        if self.session_start_time is None:
            self.session_start_time = current_session_time
            print(f"[HR INIT] Session start time set to {self.session_start_time:.1f}s")
        
        # FIXED: In real-time, the streamer gives us a growing cumulative array
        # We need to detect and add only new samples
        
        current_array_length = len(new_hr_data)
        new_samples_count = current_array_length - self.total_samples_received
        
        if new_samples_count > 0:
            # Get only the new samples from the end of the array
            new_samples = new_hr_data[-new_samples_count:] if new_samples_count < current_array_length else new_hr_data
            
            print(f"[HR BUFFER] Processing {new_samples_count} new HR samples (total array: {current_array_length})")
            
            # FIXED: Calculate proper timestamps for new samples
            for i, hr_value in enumerate(new_samples):
                # Calculate the actual timestamp for this sample based on its position
                sample_index = self.total_samples_received + i
                # Use current session time to anchor the timestamps
                sample_timestamp = current_session_time - (len(new_samples) - 1 - i) * self.hr_sample_interval
                
                # Ensure timestamp is non-negative
                sample_timestamp = max(0, sample_timestamp)
                
                # Add to buffer (we'll filter during epoch processing)
                self.hr_values.append(hr_value)
                self.hr_timestamps.append(sample_timestamp)
                
                if hr_value >= 30 and hr_value <= 200:  # Valid range
                    print(f"[HR BUFFER] Added valid HR: {hr_value:.1f} bpm at {sample_timestamp:.1f}s")
                else:
                    print(f"[HR BUFFER] Added invalid HR: {hr_value:.1f} bpm at {sample_timestamp:.1f}s")
            
            self.total_samples_received = current_array_length
            
        elif new_samples_count == 0:
            print(f"[HR BUFFER] No new samples (array length unchanged: {current_array_length})")
        else:
            # Array got smaller? This shouldn't happen in normal operation
            print(f"[HR BUFFER] Warning: Array length decreased ({current_array_length} < {self.total_samples_received})")
            print(f"[HR BUFFER] Resetting buffer with current data")
            
            # Reset and re-process all data
            self.hr_values.clear()
            self.hr_timestamps.clear()
            self.total_samples_received = 0
            
            # Re-process all samples
            for i, hr_value in enumerate(new_hr_data):
                sample_timestamp = current_session_time - (len(new_hr_data) - 1 - i) * self.hr_sample_interval
                sample_timestamp = max(0, sample_timestamp)
                
                self.hr_values.append(hr_value)
                self.hr_timestamps.append(sample_timestamp)
            
            self.total_samples_received = len(new_hr_data)
        
        # Debug info
        if len(self.hr_values) > 0:
            valid_hrs = [hr for hr in self.hr_values if 30 <= hr <= 200]
            print(f"[HR BUFFER] Total: {len(self.hr_values)} samples, valid: {len(valid_hrs)}")
            if valid_hrs:
                print(f"[HR BUFFER] Latest valid HR: {valid_hrs[-1]:.1f} bpm")

    def update_hr_buffer_batch(self, hr_array, session_start_time):
        """Update buffer with batch data (for offline-style processing)
        
        Args:
            hr_array: Complete HR array from offline processing
            session_start_time: Start time of the session
        """
        self.hr_values.clear()
        self.hr_timestamps.clear()
        self.session_start_time = session_start_time
        
        for i, hr_value in enumerate(hr_array):
            timestamp = session_start_time + i * self.hr_sample_interval
            self.hr_values.append(hr_value)
            self.hr_timestamps.append(timestamp)
        
        self.total_samples_received = len(hr_array)
        print(f"[HR BATCH] Loaded {len(hr_array)} HR samples into buffer")

    def get_buffer_status(self):
        """Get current buffer status for debugging"""
        if len(self.hr_values) == 0:
            return {
                'buffer_size': 0,
                'time_range': None,
                'latest_hr': None,
                'hr_range': None
            }
        
        valid_hr = [hr for hr in self.hr_values if 30 <= hr <= 200]
        
        return {
            'buffer_size': len(self.hr_values),
            'time_range': f"{self.hr_timestamps[0]:.1f} - {self.hr_timestamps[-1]:.1f}s" if len(self.hr_timestamps) > 0 else None,
            'latest_hr': self.hr_values[-1] if len(self.hr_values) > 0 else None,
            'hr_range': f"{min(valid_hr):.1f} - {max(valid_hr):.1f} bpm" if valid_hr else "No valid HR",
            'valid_samples': len(valid_hr),
            'total_samples': len(self.hr_values)
        }

    def get_hr_window_for_epoch(self, epoch_start_s, epoch_duration_s=4.0):
        """FINAL FIXED: Get HR data for a specific epoch time window with proper alignment
        
        Args:
            epoch_start_s: Start time of epoch in seconds (absolute session time)
            epoch_duration_s: Duration of epoch in seconds
            
        Returns:
            numpy array of HR values in the time window
        """
        if len(self.hr_values) == 0:
            print(f"   [HR WINDOW] No HR data in buffer")
            return np.array([])
        
        epoch_end_s = epoch_start_s + epoch_duration_s
        
        print(f"   [HR WINDOW] Looking for epoch {epoch_start_s:.1f}-{epoch_end_s:.1f}s")
        print(f"   [HR WINDOW] Available time range: {self.hr_timestamps[0]:.1f}-{self.hr_timestamps[-1]:.1f}s")
        
        # FIXED: Find HR samples that overlap with the epoch window
        hr_window = []
        matched_timestamps = []
        matched_indices = []
        
        for i, timestamp in enumerate(self.hr_timestamps):
            # Each HR sample represents a 5-second measurement centered on the timestamp
            # So it's valid for the time window [timestamp - 2.5, timestamp + 2.5]
            hr_sample_start = timestamp - self.hr_sample_interval / 2
            hr_sample_end = timestamp + self.hr_sample_interval / 2
            
            # Check if this HR sample overlaps with the epoch window
            if (hr_sample_start < epoch_end_s and hr_sample_end > epoch_start_s):
                hr_window.append(self.hr_values[i])
                matched_timestamps.append(timestamp)
                matched_indices.append(i)
        
        print(f"   [HR WINDOW] Found {len(hr_window)} HR samples")
        if matched_timestamps:
            print(f"   [HR WINDOW] Timestamps: {[f'{t:.1f}' for t in matched_timestamps]}")
            print(f"   [HR WINDOW] HR values: {[f'{hr:.1f}' for hr in hr_window]}")
            print(f"   [HR WINDOW] Buffer indices: {matched_indices}")
        
        return np.array(hr_window)

    def process_hr_for_epoch(self, epoch_start_s, epoch_duration_s=4.0):
        """FINAL FIXED: Extract HR features for a specific epoch (matches offline logic)
        
        Args:
            epoch_start_s: Start time of epoch in seconds (absolute session time)
            epoch_duration_s: Duration of epoch in seconds
            
        Returns:
            Dictionary of HR features
        """
        print(f"   [HR EPOCH] Processing epoch {epoch_start_s:.1f}-{epoch_start_s + epoch_duration_s:.1f}s")
        
        # Get HR window for this epoch
        hr_window = self.get_hr_window_for_epoch(epoch_start_s, epoch_duration_s)
        
        if len(hr_window) == 0:
            print(f"   [HR EPOCH] No HR data available for epoch")
            return {
                'heart_rate_87': np.nan,
                'heart_rate_88': np.nan,
                'hr_min': np.nan,
                'hr_max': np.nan,
                'hr_std': np.nan
            }
        
        # FIXED: Better filtering of invalid HR values
        # Filter out physiologically impossible values
        valid_hr_mask = (hr_window >= 30) & (hr_window <= 200)
        valid_hr = hr_window[valid_hr_mask]
        
        print(f"   [HR EPOCH] HR quality: {len(valid_hr)}/{len(hr_window)} valid samples (30-200 bpm)")
        
        if len(valid_hr) == 0:
            print(f"   [HR EPOCH] All HR samples are physiologically invalid")
            return {
                'heart_rate_87': np.nan,
                'heart_rate_88': np.nan,
                'hr_min': np.nan,
                'hr_max': np.nan,
                'hr_std': np.nan
            }
        
        # Calculate HR features (matches offline feature extraction)
        try:
            mean_hr = np.mean(valid_hr)
            min_hr = np.min(valid_hr)
            max_hr = np.max(valid_hr)
            std_hr = np.std(valid_hr) if len(valid_hr) > 1 else 0.0
            
            hr_features = {
                'heart_rate_87': mean_hr,  # Primary HR feature
                'heart_rate_88': mean_hr,  # Duplicate (as in offline)
                'hr_min': min_hr,
                'hr_max': max_hr,
                'hr_std': std_hr
            }
            
            print(f"   [HR EPOCH] ✅ Extracted HR features:")
            for name, value in hr_features.items():
                print(f"      {name}: {value:.2f}")
            
            return hr_features
            
        except Exception as e:
            print(f"   [HR EPOCH] ❌ Feature calculation failed: {e}")
            return {
                'heart_rate_87': np.nan,
                'heart_rate_88': np.nan,
                'hr_min': np.nan,
                'hr_max': np.nan,
                'hr_std': np.nan
            }

    def get_recent_hr_stats(self, duration_s=60):
        """Get HR statistics for recent time period"""
        if len(self.hr_values) == 0:
            return None
        
        current_time = self.hr_timestamps[-1]
        cutoff_time = current_time - duration_s
        
        recent_hr = []
        for i, timestamp in enumerate(self.hr_timestamps):
            if timestamp >= cutoff_time:
                hr_value = self.hr_values[i]
                if 30 <= hr_value <= 200:  # Valid HR range
                    recent_hr.append(hr_value)
        
        if len(recent_hr) == 0:
            return None
        
        return {
            'sample_count': len(recent_hr),
            'mean_hr': np.mean(recent_hr),
            'min_hr': np.min(recent_hr),
            'max_hr': np.max(recent_hr),
            'std_hr': np.std(recent_hr),
            'duration_s': duration_s
        }
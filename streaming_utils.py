# streaming_utils.py - Utilities for handling streaming data windows

import numpy as np
import time

def get_latest_window_from_buffer(buffer_data, window_samples, step_samples=None):
    """Extract the latest window from a cumulative buffer
    
    Args:
        buffer_data: Cumulative buffer (channels, total_samples) or (total_samples,)
        window_samples: Number of samples needed for the window
        step_samples: Step size for overlapping windows (optional)
    
    Returns:
        Latest window of data or None if insufficient samples
    """
    if buffer_data is None or buffer_data.size == 0:
        return None
    
    # Get total samples available
    total_samples = buffer_data.shape[1] if buffer_data.ndim == 2 else len(buffer_data)
    
    if total_samples < window_samples:
        print(f"[BUFFER] Insufficient data: {total_samples} < {window_samples} needed")
        return None
    
    # Extract the most recent window
    if buffer_data.ndim == 2:
        # Multi-channel: (channels, samples)
        latest_window = buffer_data[:, -window_samples:]
    else:
        # Single channel: (samples,)
        latest_window = buffer_data[-window_samples:]
    
    return latest_window

def get_streaming_data_with_windowing(streamer, eeg_window_samples=500, ppg_window_samples=100, imu_window_samples=None):
    """Get properly windowed data from cumulative buffers
    
    Args:
        streamer: FRENZ streamer object
        eeg_window_samples: Window size for EEG (default: 500 = 4s at 125Hz)
        ppg_window_samples: Window size for PPG (default: 100 = 4s at 25Hz)
        imu_window_samples: Window size for IMU (default: matches EEG)
    
    Returns:
        Dictionary with windowed data for each modality
    """
    data = {}
    
    if imu_window_samples is None:
        imu_window_samples = eeg_window_samples
    
    # Option 1: Use filtered EEG (already 4 channels)
    filtered_eeg = streamer.DATA["FILTERED"]["EEG"]
    if filtered_eeg is not None and filtered_eeg.size > 0:
        # Extract latest window from cumulative buffer
        eeg_window = get_latest_window_from_buffer(filtered_eeg, eeg_window_samples)
        if eeg_window is not None:
            data['eeg'] = np.array(eeg_window).copy()
            print(f"[EEG] Extracted window shape: {data['eeg'].shape} from buffer shape: {filtered_eeg.shape}")
        else:
            data['eeg'] = None
    else:
        # Fallback to raw EEG
        raw_eeg = streamer.DATA["RAW"]["EEG"]
        if raw_eeg is not None and raw_eeg.shape[0] >= eeg_window_samples:
            # Get latest window and select active channels
            raw_window = raw_eeg[-eeg_window_samples:, :]  # (samples, 6)
            # Select active channels [0, 1, 3, 4] (skip dead channel 2) and transpose
            data['eeg'] = raw_window[:, [0, 1, 3, 4]].T  # (4, samples)
            print(f"[EEG] Used raw EEG, selected 4 channels: {data['eeg'].shape}")
        else:
            data['eeg'] = None
            if raw_eeg is not None:
                print(f"[EEG] Insufficient raw data: {raw_eeg.shape[0]} < {eeg_window_samples}")
    
    # Handle PPG with initialization check
    ppg = streamer.DATA["RAW"]["PPG"]
    if ppg is not None and ppg.shape[0] >= ppg_window_samples:
        # Get latest window from GREEN channel (index 0)
        ppg_window = ppg[-ppg_window_samples:, 0]  # Green channel
        data['ppg'] = ppg_window.copy()
        print(f"[PPG] Extracted window: {len(data['ppg'])} samples")
    else:
        data['ppg'] = None
        if ppg is not None and ppg.shape[0] > 0:
            print(f"[PPG] Insufficient data: {ppg.shape[0]} < {ppg_window_samples}")
    
    # Handle IMU (only 3 channels available, need to pad to 6)
    imu = streamer.DATA["RAW"]["IMU"]
    if imu is not None and imu.shape[0] >= imu_window_samples:
        # Get latest window
        imu_window = imu[-imu_window_samples:, :].T  # (3, samples)
        # Pad to 6 channels (add zeros for missing gyroscope)
        imu_padded = np.zeros((6, imu_window_samples))
        imu_padded[0:3, :] = imu_window
        data['imu'] = imu_padded
        print(f"[IMU] Padded to 6 channels: {data['imu'].shape}")
    else:
        data['imu'] = None
        if imu is not None and imu.shape[0] > 0:
            print(f"[IMU] Insufficient data: {imu.shape[0]} < {imu_window_samples}")
    
    # Get SQC scores (current values, not windowed)
    sqc = streamer.SCORES.get("sqc_scores")
    if sqc is not None:
        data['sqc'] = np.array(sqc).copy()
    else:
        data['sqc'] = None
    
    return data

def wait_for_buffer_initialization(streamer, min_duration_s=10.0, check_interval_s=1.0):
    """Wait for streaming buffers to accumulate sufficient data
    
    Args:
        streamer: FRENZ streamer object
        min_duration_s: Minimum session duration before processing
        check_interval_s: How often to check buffer status
    
    Returns:
        True when ready, False if interrupted
    """
    print(f"[INIT] Waiting for {min_duration_s}s of data accumulation...")
    
    while streamer.session_dur < min_duration_s:
        # Check if we have any data yet
        eeg = streamer.DATA.get("FILTERED", {}).get("EEG")
        ppg = streamer.DATA.get("RAW", {}).get("PPG")
        
        # Handle different possible shapes for EEG
        eeg_samples = 0
        if eeg is not None and hasattr(eeg, 'shape'):
            if eeg.ndim == 2 and len(eeg.shape) >= 2:
                eeg_samples = eeg.shape[1]
            elif eeg.ndim == 1:
                eeg_samples = len(eeg)
            elif hasattr(eeg, 'size'):
                eeg_samples = eeg.size
        
        # Handle PPG shape
        ppg_samples = 0
        if ppg is not None and hasattr(ppg, 'shape'):
            if ppg.ndim >= 1 and len(ppg.shape) >= 1:
                ppg_samples = ppg.shape[0]
            elif hasattr(ppg, 'size'):
                ppg_samples = ppg.size
        
        print(f"[BUFFER] Duration: {streamer.session_dur:.1f}s / {min_duration_s}s | "
              f"EEG: {eeg_samples} samples | PPG: {ppg_samples} samples")
        
        time.sleep(check_interval_s)
    
    print(f"[INIT] Buffer ready! Session duration: {streamer.session_dur:.1f}s")
    return True

def validate_window_shapes(data, expected_eeg_window=500, expected_channels=4):
    """Validate that windowed data has expected shapes
    
    Args:
        data: Dictionary from get_streaming_data_with_windowing
        expected_eeg_window: Expected EEG window size
        expected_channels: Expected number of EEG channels
    
    Returns:
        Tuple (is_valid, error_message)
    """
    if data.get('eeg') is None:
        return False, "No EEG data available"
    
    eeg_shape = data['eeg'].shape
    if eeg_shape[0] != expected_channels:
        return False, f"EEG has {eeg_shape[0]} channels, expected {expected_channels}"
    
    if eeg_shape[1] != expected_eeg_window:
        return False, f"EEG has {eeg_shape[1]} samples, expected {expected_eeg_window}"
    
    # Check for NaN/Inf in EEG
    if np.any(np.isnan(data['eeg'])) or np.any(np.isinf(data['eeg'])):
        return False, "EEG data contains NaN or Inf values"
    
    # PPG is optional but validate if present
    if data.get('ppg') is not None:
        if np.any(np.isnan(data['ppg'])) or np.any(np.isinf(data['ppg'])):
            return False, "PPG data contains NaN or Inf values"
    
    return True, "Data validation passed"
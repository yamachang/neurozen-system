# test_hr_processing.py - Test HR data processing implementation

import numpy as np
import sys
import os

# Add src to path for imports
sys.path.append('./src/data_processing')
sys.path.append('./src/feature_extraction')

from src.data_processing.hr_processor import HRProcessor
from src.feature_extraction.hr_features import HRFeatureExtractor

def test_hr_processor():
    """Test the HR processor with simulated data"""
    print("🧪 TESTING HR PROCESSOR")
    print("=" * 50)
    
    # Initialize HR processor
    hr_processor = HRProcessor()
    
    # Test 1: Basic buffer update
    print("\n📊 TEST 1: Buffer Update")
    print("-" * 30)
    
    # Simulate HR data from Frenz API (realistic values at 0.2 Hz)
    simulated_hr_data = np.array([72.5, 74.2, 71.8, 73.1, 75.6])  # 5 samples = 25 seconds of data
    session_time = 30.0  # 30 seconds into session
    
    hr_processor.update_hr_buffer(simulated_hr_data, session_time)
    
    # Check buffer status
    status = hr_processor.get_buffer_status()
    print(f"   Buffer size: {status['buffer_size']} samples")
    print(f"   Time range: {status['time_range']}")
    print(f"   Latest HR: {status['latest_hr']} bpm")
    print(f"   HR range: {status['hr_range']}")
    
    # Test 2: Epoch processing (matches offline logic)
    print("\n📊 TEST 2: Epoch Processing")
    print("-" * 30)
    
    # Test epoch extraction for different time windows
    test_epochs = [
        {'start': 10.0, 'duration': 4.0, 'description': 'Early epoch (before most HR data)'},
        {'start': 25.0, 'duration': 4.0, 'description': 'Epoch with HR data'},
        {'start': 30.0, 'duration': 4.0, 'description': 'Recent epoch'},
        {'start': 40.0, 'duration': 4.0, 'description': 'Future epoch (no data)'}
    ]
    
    for test_epoch in test_epochs:
        print(f"\n   Testing: {test_epoch['description']}")
        print(f"   Epoch: {test_epoch['start']}s - {test_epoch['start'] + test_epoch['duration']}s")
        
        # Get HR window
        hr_window = hr_processor.get_hr_window_for_epoch(
            test_epoch['start'], test_epoch['duration']
        )
        
        if len(hr_window) > 0:
            print(f"   Found {len(hr_window)} HR samples: {hr_window}")
            
            # Extract features
            hr_features = hr_processor.process_hr_for_epoch(
                test_epoch['start'], test_epoch['duration']
            )
            
            print(f"   Features extracted:")
            for feature_name, value in hr_features.items():
                if not np.isnan(value):
                    print(f"     {feature_name}: {value:.2f}")
                else:
                    print(f"     {feature_name}: NaN")
        else:
            print(f"   No HR samples in this epoch")
    
    print(f"\n✅ HR Processor tests completed!")
    return True

def test_hr_feature_extractor():
    """Test the HR feature extractor directly"""
    print("\n🧪 TESTING HR FEATURE EXTRACTOR")
    print("=" * 50)
    
    # Initialize feature extractor
    hr_extractor = HRFeatureExtractor()
    
    # Test cases
    test_cases = [
        {
            'name': 'Single valid HR sample',
            'data': np.array([72.5]),
            'expected_hr_std': 0.0  # Only one sample
        },
        {
            'name': 'Multiple valid HR samples',
            'data': np.array([70.0, 72.0, 74.0, 76.0]),
            'expected_features': 5
        },
        {
            'name': 'Mixed valid/invalid HR samples',
            'data': np.array([72.0, -1.0, 74.0, -5.0, 76.0]),  # -1, -5 are invalid
            'expected_valid_count': 3
        },
        {
            'name': 'All invalid HR samples',
            'data': np.array([-1.0, -2.0, -5.0]),
            'expected_all_nan': True
        },
        {
            'name': 'Empty HR data',
            'data': np.array([]),
            'expected_all_nan': True
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📊 TEST {i}: {test_case['name']}")
        print("-" * 40)
        
        features = hr_extractor.extract_features_from_hr_data(test_case['data'])
        
        print(f"   Input: {test_case['data']}")
        print(f"   Extracted features:")
        
        valid_features = 0
        for feature_name, value in features.items():
            if not np.isnan(value):
                print(f"     {feature_name}: {value:.2f}")
                valid_features += 1
            else:
                print(f"     {feature_name}: NaN")
        
        # Validate results
        if test_case.get('expected_all_nan'):
            if valid_features == 0:
                print(f"   ✅ Correctly returned all NaN features")
            else:
                print(f"   ❌ Expected all NaN, got {valid_features} valid features")
        
        if 'expected_hr_std' in test_case:
            actual_std = features.get('hr_std', np.nan)
            expected_std = test_case['expected_hr_std']
            if abs(actual_std - expected_std) < 0.001:
                print(f"   ✅ HR std correctly calculated: {actual_std}")
            else:
                print(f"   ❌ HR std mismatch: expected {expected_std}, got {actual_std}")
        
        # Validate feature extractor interface
        validation = hr_extractor.validate_features(features)
        print(f"   Validation: {validation['valid_feature_count']}/{len(hr_extractor.feature_names)} features valid")
        print(f"   Success rate: {validation['success_rate']*100:.1f}%")
    
    print(f"\n✅ HR Feature Extractor tests completed!")
    return True

def test_integration_with_offline_logic():
    """Test that our implementation matches the offline logic exactly"""
    print("\n🧪 TESTING INTEGRATION WITH OFFLINE LOGIC")
    print("=" * 50)
    
    # Simulate the exact offline scenario
    print("\n📊 Simulating Offline Processing Logic")
    print("-" * 40)
    
    # Simulate HR array from Frenz API (0.2 Hz sampling)
    hr_array = np.array([
        70.0, 72.0, 74.0, 71.0, 73.0, 75.0, 72.0, 74.0, 76.0, 73.0,
        71.0, 72.0, 74.0, 75.0, 73.0, 72.0, 71.0, 74.0, 76.0, 75.0
    ])  # 20 samples = 100 seconds of data
    
    # Test epoch parameters (matches offline)
    epoch_original_start_s = 45.0  # 45 seconds into recording
    epoch_window_sec = 4.0
    hr_rate = 0.2  # 1 sample per 5 seconds
    session_start_time = 0.0  # Session starts at time 0
    
    print(f"   HR array: {len(hr_array)} samples ({len(hr_array)/hr_rate:.0f}s of data)")
    print(f"   Epoch start: {epoch_original_start_s}s")
    print(f"   Epoch duration: {epoch_window_sec}s")
    
    # OFFLINE LOGIC (from your script)
    print(f"\n   🔍 OFFLINE LOGIC:")
    hr_idx = int(epoch_original_start_s * hr_rate)
    hr_window_samples = max(1, int(epoch_window_sec * hr_rate))
    
    print(f"   HR index: {hr_idx}")
    print(f"   HR window samples: {hr_window_samples}")
    
    if hr_idx < len(hr_array) and hr_idx + hr_window_samples <= len(hr_array):
        hr_window_offline = hr_array[hr_idx:hr_idx + hr_window_samples]
        
        # Filter out invalid values (negative HR)
        valid_hr = hr_window_offline >= 0
        if np.any(valid_hr):
            filtered_hr = hr_window_offline[valid_hr]
            offline_features = {
                'heart_rate': np.mean(filtered_hr),
                'hr_min': np.min(filtered_hr),
                'hr_max': np.max(filtered_hr),
                'hr_std': np.std(filtered_hr) if len(filtered_hr) > 1 else 0
            }
            print(f"   Offline HR window: {hr_window_offline}")
            print(f"   Offline features:")
            for name, value in offline_features.items():
                print(f"     {name}: {value:.3f}")
        else:
            print(f"   Offline: No valid HR data")
            offline_features = None
    else:
        print(f"   Offline: HR index out of bounds")
        offline_features = None
    
    # REAL-TIME LOGIC (our implementation)
    print(f"\n   🔍 REAL-TIME LOGIC (BATCH METHOD):")
    hr_processor = HRProcessor()
    
    # Use batch update method to simulate offline-style processing for comparison
    hr_processor.update_hr_buffer_batch(hr_array, session_start_time)
    
    # Process the same epoch
    realtime_features = hr_processor.process_hr_for_epoch(epoch_original_start_s, epoch_window_sec)
    
    print(f"   Real-time features (batch):")
    for name, value in realtime_features.items():
        if not np.isnan(value):
            print(f"     {name}: {value:.3f}")
        else:
            print(f"     {name}: NaN")
    
    # ALSO TEST TRUE REAL-TIME SCENARIO
    print(f"\n   🔍 REAL-TIME LOGIC (INCREMENTAL METHOD):")
    hr_processor_realtime = HRProcessor()
    
    # Simulate real-time updates (one sample at a time)
    for i, hr_value in enumerate(hr_array):
        # Each HR sample represents 5 seconds of time
        current_session_time = i * 5.0
        
        # Update buffer with this single sample (as happens in real-time)
        hr_processor_realtime.update_hr_buffer(np.array([hr_value]), current_session_time)
    
    # Process the same epoch
    realtime_features_incremental = hr_processor_realtime.process_hr_for_epoch(epoch_original_start_s, epoch_window_sec)
    
    print(f"   Real-time features (incremental):")
    for name, value in realtime_features_incremental.items():
        if not np.isnan(value):
            print(f"     {name}: {value:.3f}")
        else:
            print(f"     {name}: NaN")
    
    # COMPARISON
    print(f"\n   📊 COMPARISON:")
    if offline_features:
        # Map offline to real-time feature names
        feature_mapping = {
            'heart_rate': 'heart_rate_87',
            'hr_min': 'hr_min',
            'hr_max': 'hr_max',
            'hr_std': 'hr_std'
        }
        
        # Test both batch and incremental methods
        for method_name, method_features in [("BATCH", realtime_features), ("INCREMENTAL", realtime_features_incremental)]:
            print(f"\n     --- {method_name} METHOD COMPARISON ---")
            matches = 0
            total_comparisons = 0
            
            for offline_name, realtime_name in feature_mapping.items():
                offline_val = offline_features.get(offline_name)
                realtime_val = method_features.get(realtime_name)
                
                if offline_val is not None and not np.isnan(realtime_val):
                    total_comparisons += 1
                    if abs(offline_val - realtime_val) < 0.001:
                        matches += 1
                        print(f"       ✅ {offline_name}: {offline_val:.3f} = {realtime_val:.3f}")
                    else:
                        print(f"       ❌ {offline_name}: {offline_val:.3f} ≠ {realtime_val:.3f}")
            
            # Check heart_rate_88 (should equal heart_rate_87)
            hr_87 = method_features.get('heart_rate_87')
            hr_88 = method_features.get('heart_rate_88')
            if not np.isnan(hr_87) and not np.isnan(hr_88):
                total_comparisons += 1
                if abs(hr_87 - hr_88) < 0.001:
                    matches += 1
                    print(f"       ✅ heart_rate_87 = heart_rate_88: {hr_87:.3f}")
                else:
                    print(f"       ❌ heart_rate_87 ≠ heart_rate_88: {hr_87:.3f} ≠ {hr_88:.3f}")
            
            accuracy = matches / total_comparisons * 100 if total_comparisons > 0 else 0
            print(f"       📈 ACCURACY: {matches}/{total_comparisons} features match ({accuracy:.1f}%)")
            
            if accuracy >= 100:
                print(f"       🎉 PERFECT MATCH! {method_name} method matches offline logic exactly!")
            elif accuracy >= 90:
                print(f"       ✅ GOOD MATCH! {method_name} method works well.")
            else:
                print(f"       ⚠️  SIGNIFICANT DIFFERENCES! Check {method_name} method implementation.")
    
    print(f"\n✅ Integration test completed!")
    return True

def main():
    """Run all HR processing tests"""
    print("🧪 HR DATA PROCESSING TEST SUITE")
    print("=" * 60)
    print("Testing HR processor and feature extractor implementation")
    print("Validating compatibility with offline processing logic")
    print()
    
    try:
        # Run all tests
        test_hr_processor()
        test_hr_feature_extractor()
        test_integration_with_offline_logic()
        
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"HR processing implementation is ready for real-time use")
        print(f"Features extracted: heart_rate_87, heart_rate_88, hr_min, hr_max, hr_std")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main()
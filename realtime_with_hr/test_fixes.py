# test_fixes.py - Test script to verify all fixes work correctly

import numpy as np
import json
import os

def test_normalization_loading():
    """Test that normalization stats can be loaded correctly"""
    print("🧪 TESTING NORMALIZATION LOADING")
    print("=" * 50)
    
    # Test with your actual normalization stats file
    norm_stats_path = "./data/processed/ml_normalization_stats.json"
    
    if not os.path.exists(norm_stats_path):
        print(f"❌ Normalization stats file not found: {norm_stats_path}")
        print("   Please make sure you have run the offline processing script")
        return False
    
    try:
        with open(norm_stats_path, 'r') as f:
            loaded_data = json.load(f)
        
        print(f"✅ Successfully loaded normalization file")
        print(f"   File size: {os.path.getsize(norm_stats_path)} bytes")
        
        # Check structure
        if 'feature_stats' in loaded_data:
            feature_stats = loaded_data['feature_stats']
            print(f"   Features in stats: {len(feature_stats)}")
            
            # Check for key features
            key_features = ['LF_rel_theta_power', 'RF_rel_alpha_power', 'LF_rel_alpha_power']
            found_features = [f for f in key_features if f in feature_stats]
            print(f"   Key features found: {found_features}")
            
            # Show sample feature
            if found_features:
                sample_feature = found_features[0]
                sample_stats = feature_stats[sample_feature]
                print(f"   Sample feature '{sample_feature}':")
                print(f"     Mean: {sample_stats.get('mean', 'N/A')}")
                print(f"     Std: {sample_stats.get('std', 'N/A')}")
                print(f"     Initialized: {sample_stats.get('initialized', 'N/A')}")
            
            return True
        else:
            print(f"❌ Unexpected structure in normalization file")
            print(f"   Top-level keys: {list(loaded_data.keys())}")
            return False
            
    except Exception as e:
        print(f"❌ Error loading normalization stats: {e}")
        return False

def test_hr_processor():
    """Test HR processor with simulated real-time data"""
    print("\n🧪 TESTING HR PROCESSOR")
    print("=" * 50)
    
    # Import the fixed HR processor
    try:
        from src.data_processing.hr_processor import HRProcessor
    except ImportError:
        print("❌ Could not import HRProcessor - check import path")
        return False
    
    # Initialize processor
    hr_processor = HRProcessor()
    
    # FIXED: Simulate real-time HR data updates correctly (cumulative growing arrays)
    print("\n📊 Simulating real-time HR updates (cumulative arrays)...")
    
    # Test data similar to your session output - simulate how streamer actually works
    test_hr_updates = [
        (np.array([-1]), 10.0),                           # First sample
        (np.array([-1, 0]), 15.0),                        # Growing array  
        (np.array([-1, 0, 72]), 20.0),                    # Valid data appears
        (np.array([-1, 0, 72, 74]), 25.0),                # More valid data
        (np.array([-1, 0, 72, 74, 76]), 30.0),            # Even more data
        (np.array([-1, 0, 72, 74, 76, 75]), 35.0),        # Continue growing
        (np.array([-1, 0, 72, 74, 76, 75, 78]), 40.0),    # More valid data
        (np.array([-1, 0, 72, 74, 76, 75, 78, 80]), 45.0), # Latest data
    ]
    
    for hr_data, session_time in test_hr_updates:
        print(f"\n--- Updating HR buffer at {session_time:.1f}s ---")
        print(f"Input HR data: {hr_data} (length: {len(hr_data)})")
        
        hr_processor.update_hr_buffer(hr_data, session_time)
        
        # Check buffer status
        status = hr_processor.get_buffer_status()
        print(f"Buffer status: {status['buffer_size']} samples, {status['valid_samples']} valid")
        if status['latest_hr']:
            print(f"Latest HR: {status['latest_hr']:.1f} bpm")
    
    # Test epoch processing at different times
    print(f"\n📊 Testing epoch processing...")
    test_epochs = [
        (10.0, "Early epoch - should have limited data"),
        (25.0, "Mid epoch - should have some valid data"), 
        (40.0, "Late epoch - should have good data"),
    ]
    
    for epoch_start, description in test_epochs:
        print(f"\n--- Testing {description} ---")
        print(f"Epoch: {epoch_start:.1f}s - {epoch_start + 4.0:.1f}s")
        
        hr_features = hr_processor.process_hr_for_epoch(epoch_start, 4.0)
        
        valid_features = [(k, v) for k, v in hr_features.items() if not np.isnan(v)]
        print(f"Results: {len(valid_features)}/5 valid features")
        if valid_features:
            print(f"Sample values: {valid_features[:3]}")
    
    return True

def test_feature_normalization():
    """Test that feature normalization creates smoothed features"""
    print("\n🧪 TESTING FEATURE NORMALIZATION")
    print("=" * 50)
    
    # FIXED: Use the correct feature names that match the normalization stats (uppercase)
    sample_features = {
        'LF_abs_theta_power': 25.0,      # Match normalization stats case
        'LF_rel_theta_power': 0.25,      # Match normalization stats case
        'LF_abs_alpha_power': 12.0,      # Match normalization stats case
        'LF_rel_alpha_power': 0.12,      # Match normalization stats case
        'RF_abs_theta_power': 1.5,       # Match normalization stats case
        'RF_rel_theta_power': 0.20,      # Match normalization stats case
        'RF_abs_alpha_power': 0.8,       # Match normalization stats case
        'RF_rel_alpha_power': 0.10,      # Match normalization stats case
        'heart_rate_87': np.nan,         # Simulate missing HR
        'stillness_score': 0.8,          # IMU feature
    }
    
    print(f"Input features: {len(sample_features)}")
    for name, value in sample_features.items():
        if not np.isnan(value):
            print(f"  {name}: {value}")
        else:
            print(f"  {name}: NaN")
    
    # Try to import and test the processor
    try:
        from realtime_processor import FixedNormalizationRealTimeProcessor
        
        # Test with correct path
        norm_path = "./data/processed/ml_normalization_stats.json"
        processor = FixedNormalizationRealTimeProcessor(
            normalization_stats_path=norm_path
        )
        
        print(f"\nProcessor initialized:")
        print(f"  Normalization loaded: {processor.normalization_loaded}")
        print(f"  Normalization stats count: {len(processor.normalization_stats)}")
        
        # Debug: Show some normalization stat keys
        if processor.normalization_stats:
            norm_keys = list(processor.normalization_stats.keys())[:5]
            print(f"  Sample normalization keys: {norm_keys}")
        
        # Apply normalization
        normalized_features = processor._apply_fixed_normalization(sample_features)
        
        print(f"\nNormalization results:")
        print(f"  Input features: {len(sample_features)}")
        print(f"  Output features: {len(normalized_features)}")
        
        # Count smoothed features
        smoothed_features = [k for k in normalized_features.keys() if k.endswith('_smoothed')]
        print(f"  Smoothed features: {len(smoothed_features)}")
        
        if smoothed_features:
            print(f"  Smoothed feature examples: {smoothed_features[:3]}")
            
            # Show some normalized values
            normalized_examples = []
            for name, value in normalized_features.items():
                if not name.endswith('_smoothed') and not np.isnan(value) and name != 'stillness_score':
                    original_value = sample_features.get(name, sample_features.get(name.upper(), 'N/A'))
                    normalized_examples.append(f"{name}: {original_value} → {value:.3f}")
                    if len(normalized_examples) >= 3:
                        break
            
            if normalized_examples:
                print(f"  Normalization examples: {normalized_examples}")
            
            return True
        else:
            print(f"  ❌ No smoothed features created!")
            
            # Debug: Check which features got normalized
            normalized_power_features = [k for k in normalized_features.keys() 
                                       if any(keyword in k for keyword in ['_abs_', '_rel_', '_power'])]
            print(f"  Normalized power features: {len(normalized_power_features)}")
            if normalized_power_features:
                print(f"  Examples: {normalized_power_features[:3]}")
            
            return False
            
    except Exception as e:
        print(f"❌ Error testing normalization: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🔧 TESTING ALL FIXES")
    print("=" * 60)
    print("This script tests the HR processing and normalization fixes")
    print()
    
    results = []
    
    # Test 1: Normalization loading
    results.append(test_normalization_loading())
    
    # Test 2: HR processor
    results.append(test_hr_processor())
    
    # Test 3: Feature normalization
    results.append(test_feature_normalization())
    
    # Summary
    print(f"\n🏁 TEST SUMMARY")
    print("=" * 30)
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Your fixes should work in the real-time session")
        print()
        print("Expected results in real-time session:")
        print("• Valid HR features extracted (heart_rate_87, heart_rate_88, etc.)")
        print("• 20 smoothed features created (_smoothed variants)")  
        print("• Complete 87-feature set (67 base + 20 smoothed)")
        print("• Normalization applied successfully")
        print("• LSTM feature alignment: 87/87")
        print()
        print("Next steps:")
        print("1. Replace realtime_processor.py with the fixed version")
        print("2. Replace src/data_processing/hr_processor.py with the fixed version")  
        print("3. Run your real-time session")
        print("4. Look for: '[PROC DETAIL] Created X smoothed features'")
        print("5. Look for: 'HR Features: X/5 valid' with actual values")
    else:
        print("⚠️  Some tests failed - check the error messages above")
        print("Common issues:")
        print("- Normalization stats file not found or wrong path")
        print("- Feature name case mismatch (uppercase vs lowercase)")
        print("- HR processor import path incorrect")
        print("- Missing offline processing step (run process_all_sessions_12.py first)")
    
    return passed == total

if __name__ == "__main__":
    main()
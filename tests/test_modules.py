# tests/test_modules.py

import unittest
import time
import numpy as np
import json
import os
import tempfile
import threading
import queue
from collections import deque
from unittest.mock import Mock, patch, MagicMock
import warnings
warnings.filterwarnings('ignore')

# Performance monitoring
import psutil
from dataclasses import dataclass
from typing import Dict, List, Optional

try:
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    print(f"Import paths:")
    print(f"   Current dir: {current_dir}")
    print(f"   Parent dir: {parent_dir}")
    print(f"   Added to sys.path: {parent_dir}")
    
    from realtime_processor import FixedNormalizationRealTimeProcessor
    from lstm_inference import AlignedLSTMInferenceEngine  
    from adaptive_audio_system import BulletproofAudioSystem, MeditationAudioManager
    from feature_alignment_fix import FeatureAligner
    from html_meditation_visualizer import HTMLMeditationVisualizer
    
    print(f"All imports successful!")
    IMPORTS_AVAILABLE = True
    
except ImportError as e:
    print(f"Import failed: {e}")
    print(f"   Current working directory: {os.getcwd()}")
    print(f"   Python path: {sys.path}")
    IMPORTS_AVAILABLE = False
except Exception as e:
    print(f"Unexpected error during imports: {e}")
    IMPORTS_AVAILABLE = False

@dataclass
class TestResult:
    """Test result container"""
    name: str
    passed: bool
    duration_ms: float
    details: Dict
    error: str = None

class MeditationSystemTester:
    """Comprehensive tester for meditation system"""
    
    def __init__(self):
        self.results = []
        self.start_time = time.time()
        
        # Performance targets
        self.performance_targets = {
            'eeg_processing': 500,      
            'lstm_inference': 100,      
            'feature_alignment': 50,    
            'audio_update': 50,         
            'visualization_update': 100 
        }
        
        # Test data directory
        self.test_data_dir = tempfile.mkdtemp()
        
        # Check if imports are available
        if not IMPORTS_AVAILABLE:
            print("WARNING: Some imports not available. Tests will be limited.")
        
    def log_result(self, name: str, passed: bool, duration_ms: float, details: Dict, error: str = None):
        """Log a test result"""
        result = TestResult(name, passed, duration_ms, details, error)
        self.results.append(result)
        
        status = "PASS" if passed else "FAIL"
        print(f"{status} {name} ({duration_ms:.1f}ms)")
        if error:
            print(f"    Error: {error}")
        if details:
            for key, value in details.items():
                print(f"    {key}: {value}")
    
    def run_all_tests(self):
        """Run comprehensive test suite"""
        print("COMPREHENSIVE MEDITATION SYSTEM TESTING")
        print("=" * 60)
        
        # Check imports first
        if not IMPORTS_AVAILABLE:
            print("Cannot run tests - required modules not importable")
            print("   Please ensure all Python files are in the correct locations")
            return self.generate_final_report()
        
        # 1. Unit Tests
        print("\n1. UNIT TESTS")
        print("-" * 20)
        self.run_unit_tests()
        
        # 2. Latency Tests  
        print("\n2. LATENCY TESTS")
        print("-" * 20)
        self.run_latency_tests()
        
        # 3. Integration Tests
        print("\n3. INTEGRATION TESTS")
        print("-" * 20)
        self.run_integration_tests()
        
        # 4. Generate Report
        print("\n4. FINAL REPORT")
        print("-" * 20)
        return self.generate_final_report()
    
    def run_unit_tests(self):
        """Run unit tests for individual components"""
        
        if not IMPORTS_AVAILABLE:
            self.log_result("Unit Test: All Tests", False, 0, {}, "Required imports not available")
            return
        
        # Test 1: Feature Alignment
        start_time = time.perf_counter()
        try:
            aligner = FeatureAligner()
            
            # Test with realistic mismatched features
            test_features = {
                'LF_ABS_ALPHA_POWER': -1.5,
                'rf_rel_theta_power': -2.1,
                'OTEL_SEF95': 15.2,
                'heart_rate_87': 72.0,
                'stillness_score': 0.8,
                'invalid_feature': 999, 
            }
            
            aligned = aligner.align_features(test_features)
            
            # Verify correct number of features
            features_correct = len(aligned) == 87
            
            # Verify feature types
            all_numeric = all(isinstance(v, (int, float)) and not np.isnan(v) 
                            for v in aligned.values())
            
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            passed = features_correct and all_numeric
            self.log_result(
                "Unit Test: Feature Alignment",
                passed,
                duration_ms,
                {
                    'input_features': len(test_features),
                    'output_features': len(aligned),
                    'target_features': 87,
                    'all_numeric': all_numeric
                }
            )
            
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            self.log_result("Unit Test: Feature Alignment", False, duration_ms, {}, str(e))
        
        # Test 2: EEG Processor Initialization
        start_time = time.perf_counter()
        try:
            processor = FixedNormalizationRealTimeProcessor()
            
            # Test initialization
            has_eeg_processor = hasattr(processor, 'eeg_processor') and processor.eeg_processor is not None
            has_feature_extractor = hasattr(processor, 'eeg_feature_extractor') and processor.eeg_feature_extractor is not None
            
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            passed = has_eeg_processor and has_feature_extractor
            self.log_result(
                "Unit Test: EEG Processor Init",
                passed,
                duration_ms,
                {
                    'eeg_processor_available': has_eeg_processor,
                    'feature_extractor_available': has_feature_extractor,
                    'normalization_loaded': processor.normalization_loaded
                }
            )
            
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            self.log_result("Unit Test: EEG Processor Init", False, duration_ms, {}, str(e))
        
        # Test 3: Audio System Mock
        start_time = time.perf_counter()
        try:
            # Test without actual hardware
            audio_system = BulletproofAudioSystem(target_device="MOCK_DEVICE")
            
            # Should handle missing device gracefully
            device_handling = audio_system.audio_device_id is None
            
            # Test parameter updates
            audio_system.update_meditation_state(1, 0.8)
            
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            self.log_result(
                "Unit Test: Audio System Mock",
                device_handling,
                duration_ms,
                {
                    'handles_missing_device': device_handling,
                    'parameter_update_successful': True
                }
            )
            
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            self.log_result("Unit Test: Audio System Mock", False, duration_ms, {}, str(e))
        
        # Test 4: Visualization System
        start_time = time.perf_counter()
        try:
            test_file = os.path.join(self.test_data_dir, "test_viz.json")
            visualizer = HTMLMeditationVisualizer(data_file=test_file)
            
            # Test adding data
            current_time = time.time()
            visualizer.add_meditation_prediction(current_time, 1, 0.8)
            
            alpha_power = np.array([-35.0, -32.0, -38.0, -33.0])
            theta_power = np.array([-40.0, -37.0, -42.0, -39.0])
            
            visualizer.add_brainwave_data_from_api(
                current_time, alpha_power, theta_power
            )
            
            # Test display update
            visualizer.update_display()
            
            # Verify data file creation
            file_exists = os.path.exists(test_file)
            
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            self.log_result(
                "Unit Test: Visualization System",
                file_exists,
                duration_ms,
                {
                    'data_file_created': file_exists,
                    'setup_success': visualizer.setup_success
                }
            )
            
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            self.log_result("Unit Test: Visualization System", False, duration_ms, {}, str(e))
    
    def run_latency_tests(self):
        """Run latency/performance tests"""
        
        if not IMPORTS_AVAILABLE:
            self.log_result("Latency Test: All Tests", False, 0, {}, "Required imports not available")
            return
        
        # Test 1: EEG Processing Latency
        print("Testing EEG Processing Latency...")
        
        try:
            processor = FixedNormalizationRealTimeProcessor()
            
            # Generate test EEG data (4 channels, 4 seconds at 125 Hz = 500 samples)
            test_eeg = np.random.randn(4, 500) * 10  
            test_sqc = np.array([1, 1, 0, 0])  # Mixed quality scores
            
            latencies = []
            test_iterations = 50
            
            for i in range(test_iterations):
                start_time = time.perf_counter()
                
                try:
                    features = processor.process_realtime_data(
                        eeg_data=test_eeg,
                        sqc_scores=test_sqc,
                        session_duration=i * 4
                    )
                    
                    end_time = time.perf_counter()
                    latency_ms = (end_time - start_time) * 1000
                    latencies.append(latency_ms)
                    
                except Exception:
                    latencies.append(float('inf'))
            
            # Calculate statistics
            valid_latencies = [l for l in latencies if l != float('inf')]
            
            if valid_latencies:
                mean_ms = np.mean(valid_latencies)
                p95_ms = np.percentile(valid_latencies, 95)
                max_ms = np.max(valid_latencies)
                success_rate = len(valid_latencies) / len(latencies) * 100
                
                target_ms = self.performance_targets['eeg_processing']
                within_target = p95_ms < target_ms
                
                self.log_result(
                    "Latency Test: EEG Processing",
                    within_target,
                    mean_ms,
                    {
                        'mean_ms': f"{mean_ms:.1f}",
                        'p95_ms': f"{p95_ms:.1f}",
                        'max_ms': f"{max_ms:.1f}",
                        'target_ms': target_ms,
                        'success_rate_percent': f"{success_rate:.1f}",
                        'samples_tested': test_iterations
                    }
                )
            else:
                self.log_result("Latency Test: EEG Processing", False, 0, {}, "All processing attempts failed")
                
        except Exception as e:
            self.log_result("Latency Test: EEG Processing", False, 0, {}, str(e))
        
        # Test 2: Feature Alignment Latency
        print("Testing Feature Alignment Latency...")
        
        try:
            # Generate test features (87 features expected)
            feature_names = [
                "lf_abs_theta_power", "lf_rel_theta_power", "lf_sef95",
                "rf_abs_theta_power", "rf_rel_theta_power", "rf_sef95",
                "heart_rate_87", "stillness_score"
            ] + [f"feature_{i}" for i in range(79)]  # Fill to 87 features
            
            latencies = []
            test_iterations = 100
            
            for i in range(test_iterations):
                # Create realistic test features
                test_features = {}
                for feature_name in feature_names:
                    if 'power' in feature_name:
                        test_features[feature_name] = np.random.normal(-35, 10)  # Log power
                    elif 'heart_rate' in feature_name:
                        test_features[feature_name] = np.random.normal(70, 10)  # BPM
                    elif 'stillness' in feature_name:
                        test_features[feature_name] = np.random.uniform(0, 1)
                    else:
                        test_features[feature_name] = np.random.normal(0, 1)
                
                start_time = time.perf_counter()
                
                try:
                    # Test feature alignment (this is what happens in real system)
                    aligner = FeatureAligner()
                    aligned_features = aligner.align_features(test_features)
                    
                    end_time = time.perf_counter()
                    
                    if len(aligned_features) == 87:
                        latency_ms = (end_time - start_time) * 1000
                        latencies.append(latency_ms)
                    else:
                        latencies.append(float('inf'))
                        
                except Exception:
                    latencies.append(float('inf'))
            
            # Calculate statistics
            valid_latencies = [l for l in latencies if l != float('inf')]
            
            if valid_latencies:
                mean_ms = np.mean(valid_latencies)
                p95_ms = np.percentile(valid_latencies, 95)
                max_ms = np.max(valid_latencies)
                success_rate = len(valid_latencies) / len(latencies) * 100
                
                target_ms = self.performance_targets['feature_alignment']
                within_target = p95_ms < target_ms
                
                self.log_result(
                    "Latency Test: Feature Alignment",
                    within_target,
                    mean_ms,
                    {
                        'mean_ms': f"{mean_ms:.1f}",
                        'p95_ms': f"{p95_ms:.1f}",
                        'max_ms': f"{max_ms:.1f}",
                        'target_ms': target_ms,
                        'success_rate_percent': f"{success_rate:.1f}",
                        'samples_tested': test_iterations
                    }
                )
            else:
                self.log_result("Latency Test: Feature Alignment", False, 0, {}, "All alignment attempts failed")
                
        except Exception as e:
            self.log_result("Latency Test: Feature Alignment", False, 0, {}, str(e))
        
        # Test 3: Audio System Latency
        print("Testing Audio System Latency...")
        
        try:
            audio_system = BulletproofAudioSystem(target_device="MOCK")
            
            # Test parameter updates
            update_latencies = []
            test_iterations = 50
            
            for i in range(test_iterations):
                start_time = time.perf_counter()
                
                # Simulate meditation state update
                test_state = i % 3  # Cycle through states 0, 1, 2
                test_confidence = 0.5 + (i % 5) * 0.1
                
                audio_system.update_meditation_state(test_state, test_confidence)
                
                end_time = time.perf_counter()
                update_latency_ms = (end_time - start_time) * 1000
                update_latencies.append(update_latency_ms)
            
            if update_latencies:
                mean_ms = np.mean(update_latencies)
                p95_ms = np.percentile(update_latencies, 95)
                max_ms = np.max(update_latencies)
                
                target_ms = self.performance_targets['audio_update']
                within_target = p95_ms < target_ms
                
                # Calculate buffer latency if available
                buffer_latency_ms = 0
                if hasattr(audio_system, 'buffer_size') and hasattr(audio_system, 'sample_rate'):
                    buffer_latency_ms = (audio_system.buffer_size / audio_system.sample_rate) * 1000
                
                self.log_result(
                    "Latency Test: Audio Updates",
                    within_target,
                    mean_ms,
                    {
                        'update_mean_ms': f"{mean_ms:.2f}",
                        'update_p95_ms': f"{p95_ms:.2f}",
                        'update_max_ms': f"{max_ms:.2f}",
                        'buffer_latency_ms': f"{buffer_latency_ms:.1f}",
                        'target_ms': target_ms,
                        'samples_tested': test_iterations
                    }
                )
            else:
                self.log_result("Latency Test: Audio Updates", False, 0, {}, "No latency data collected")
                
        except Exception as e:
            self.log_result("Latency Test: Audio Updates", False, 0, {}, str(e))
        
        # Test 4: End-to-End Pipeline Latency
        print("Testing End-to-End Pipeline...")
        
        try:
            total_latencies = []
            test_iterations = 20
            
            for i in range(test_iterations):
                start_time = time.perf_counter()
                
                # Simulate complete pipeline
                try:
                    # 1. Generate mock EEG data
                    mock_eeg = np.random.randn(4, 500) * 10
                    
                    # 2. Process data
                    processor = FixedNormalizationRealTimeProcessor()
                    features = processor.process_realtime_data(
                        eeg_data=mock_eeg,
                        sqc_scores=np.array([1, 1, 0, 0]),
                        session_duration=i * 4
                    )
                    
                    # 3. Feature alignment
                    if features:
                        aligner = FeatureAligner()
                        aligned = aligner.align_features(features[0])
                    
                    # 4. Audio update (mock)
                    audio_system = BulletproofAudioSystem(target_device="MOCK")
                    audio_system.update_meditation_state(i % 3, 0.8)
                    
                    end_time = time.perf_counter()
                    total_latency_ms = (end_time - start_time) * 1000
                    total_latencies.append(total_latency_ms)
                    
                except Exception:
                    total_latencies.append(float('inf'))
            
            # Calculate statistics
            valid_latencies = [l for l in total_latencies if l != float('inf')]
            
            if valid_latencies:
                mean_ms = np.mean(valid_latencies)
                p95_ms = np.percentile(valid_latencies, 95)
                max_ms = np.max(valid_latencies)
                success_rate = len(valid_latencies) / len(total_latencies) * 100
                
                target_ms = 1000  # 1 second total target
                within_target = p95_ms < target_ms
                
                self.log_result(
                    "Latency Test: End-to-End Pipeline",
                    within_target,
                    mean_ms,
                    {
                        'mean_ms': f"{mean_ms:.1f}",
                        'p95_ms': f"{p95_ms:.1f}",
                        'max_ms': f"{max_ms:.1f}",
                        'target_ms': target_ms,
                        'success_rate_percent': f"{success_rate:.1f}",
                        'samples_tested': test_iterations
                    }
                )
            else:
                self.log_result("Latency Test: End-to-End Pipeline", False, 0, {}, "All pipeline tests failed")
                
        except Exception as e:
            self.log_result("Latency Test: End-to-End Pipeline", False, 0, {}, str(e))
    
    def run_integration_tests(self):
        """Run integration tests for complete system"""
        
        if not IMPORTS_AVAILABLE:
            self.log_result("Integration Test: All Tests", False, 0, {}, "Required imports not available")
            return
        
        # Test 1: Processing Pipeline Integration
        start_time = time.perf_counter()
        try:
            # Initialize processor
            processor = FixedNormalizationRealTimeProcessor()
            
            # Generate realistic mock EEG data
            eeg_data = np.random.randn(4, 500) * 10  # 4 channels, 4 seconds at 125Hz
            sqc_scores = np.array([1, 1, 0, 0])  # Mixed quality
            
            # Process multiple epochs to build session buffer
            all_features = []
            
            for epoch in range(5):
                features = processor.process_realtime_data(
                    eeg_data=eeg_data + np.random.randn(*eeg_data.shape),  # Add noise
                    sqc_scores=sqc_scores,
                    session_duration=epoch * 4
                )
                
                all_features.extend(features)
            
            # Verify features were extracted
            features_extracted = len(all_features) > 0
            
            # Verify feature structure
            structure_valid = True
            if features_extracted:
                for feature_dict in all_features:
                    if 'eeg_quality_flag' not in feature_dict:
                        structure_valid = False
                        break
                    if not isinstance(feature_dict['eeg_quality_flag'], bool):
                        structure_valid = False
                        break
            
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            passed = features_extracted and structure_valid
            self.log_result(
                "Integration Test: Processing Pipeline",
                passed,
                duration_ms,
                {
                    'features_extracted': len(all_features),
                    'structure_valid': structure_valid,
                    'epochs_processed': 5
                }
            )
            
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            self.log_result("Integration Test: Processing Pipeline", False, duration_ms, {}, str(e))
        
        # Test 2: Feature to LSTM Pipeline
        start_time = time.perf_counter()
        try:
            # Create mock features
            mock_features = {}
            
            # Generate 87 features with realistic values
            feature_names = [
                "lf_abs_theta_power", "lf_rel_theta_power", "lf_sef95",
                "rf_abs_theta_power", "rf_rel_theta_power", "rf_sef95",
                "heart_rate_87", "stillness_score"
            ] + [f"mock_feature_{i}" for i in range(79)]
            
            for feature_name in feature_names:
                if 'power' in feature_name:
                    mock_features[feature_name] = np.random.normal(-35, 10)
                elif 'heart_rate' in feature_name:
                    mock_features[feature_name] = np.random.normal(70, 10)
                elif 'stillness' in feature_name:
                    mock_features[feature_name] = np.random.uniform(0, 1)
                else:
                    mock_features[feature_name] = np.random.normal(0, 1)
            
            # Test feature alignment
            aligner = FeatureAligner()
            aligned_features = aligner.align_features(mock_features)
            
            # Verify alignment
            alignment_correct = len(aligned_features) == 87
            
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            self.log_result(
                "Integration Test: Feature to LSTM Pipeline",
                alignment_correct,
                duration_ms,
                {
                    'input_features': len(mock_features),
                    'aligned_features': len(aligned_features),
                    'target_features': 87
                }
            )
            
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            self.log_result("Integration Test: Feature to LSTM Pipeline", False, duration_ms, {}, str(e))
        
        # Test 3: Complete System Integration
        start_time = time.perf_counter()
        try:
            results = {}
            
            # 1. Test processor
            processor = FixedNormalizationRealTimeProcessor()
            results['processor'] = processor is not None
            
            # 2. Test feature alignment
            aligner = FeatureAligner()
            test_features = {'test_feature': 1.0}
            aligned = aligner.align_features(test_features)
            results['feature_alignment'] = len(aligned) == 87
            
            # 3. Test visualizer
            test_file = os.path.join(self.test_data_dir, "test_integration.json")
            visualizer = HTMLMeditationVisualizer(data_file=test_file)
            current_time = time.time()
            visualizer.add_meditation_prediction(current_time, 1, 0.8)
            visualizer.update_display()
            results['visualizer'] = os.path.exists(test_file)
            
            # 4. Test audio system (mock)
            try:
                audio_system = BulletproofAudioSystem(target_device="MOCK")
                results['audio_system'] = audio_system is not None
            except:
                results['audio_system'] = False  # Expected with mock device
            
            # Verify all components
            core_components = ['processor', 'feature_alignment', 'visualizer']
            core_success = all(results.get(comp, False) for comp in core_components)
            
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            self.log_result(
                "Integration Test: Complete System",
                core_success,
                duration_ms,
                results
            )
            
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            self.log_result("Integration Test: Complete System", False, duration_ms, {}, str(e))
    
    def generate_final_report(self):
        """Generate comprehensive test report"""
        end_time = time.time()
        total_duration = end_time - self.start_time
        
        # Calculate statistics
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.passed)
        failed_tests = total_tests - passed_tests
        
        # Categorize results
        unit_tests = [r for r in self.results if "Unit Test" in r.name]
        latency_tests = [r for r in self.results if "Latency Test" in r.name]
        integration_tests = [r for r in self.results if "Integration Test" in r.name]
        
        # Performance assessment
        latency_performance = all(r.passed for r in latency_tests)
        
        # Generate report
        report = {
            'timestamp': time.time(),
            'total_duration_s': total_duration,
            'imports_available': IMPORTS_AVAILABLE,
            'summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'success_rate_percent': (passed_tests / total_tests * 100) if total_tests > 0 else 0,
                'real_time_capable': latency_performance and IMPORTS_AVAILABLE
            },
            'categories': {
                'unit_tests': {
                    'total': len(unit_tests),
                    'passed': sum(1 for r in unit_tests if r.passed),
                    'results': [{'name': r.name, 'passed': r.passed, 'duration_ms': r.duration_ms, 'details': r.details, 'error': r.error} for r in unit_tests]
                },
                'latency_tests': {
                    'total': len(latency_tests),
                    'passed': sum(1 for r in latency_tests if r.passed),
                    'real_time_capable': all(r.passed for r in latency_tests),
                    'results': [{'name': r.name, 'passed': r.passed, 'duration_ms': r.duration_ms, 'details': r.details, 'error': r.error} for r in latency_tests]
                },
                'integration_tests': {
                    'total': len(integration_tests),
                    'passed': sum(1 for r in integration_tests if r.passed),
                    'results': [{'name': r.name, 'passed': r.passed, 'duration_ms': r.duration_ms, 'details': r.details, 'error': r.error} for r in integration_tests]
                }
            },
            'performance_targets': self.performance_targets,
            'all_results': [{'name': r.name, 'passed': r.passed, 'duration_ms': r.duration_ms, 'details': r.details, 'error': r.error} for r in self.results]
        }
        
        # Print summary
        print(f"\n" + "="*60)
        print("FINAL TEST REPORT")
        print("="*60)
        
        print(f"Import Status: {'✅ SUCCESS' if IMPORTS_AVAILABLE else '❌ FAILED'}")
        print(f"Total Duration: {total_duration:.1f}s")
        print(f"Tests Run: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"Success Rate: {(passed_tests / total_tests * 100):.1f}%" if total_tests > 0 else "Success Rate: 0%")
        
        print(f"\nBy Category:")
        print(f"   Unit Tests: {sum(1 for r in unit_tests if r.passed)}/{len(unit_tests)}")
        print(f"   Latency Tests: {sum(1 for r in latency_tests if r.passed)}/{len(latency_tests)}")
        print(f"   Integration Tests: {sum(1 for r in integration_tests if r.passed)}/{len(integration_tests)}")
        
        print(f"\n🚀 Real-time Performance: {'✅ CAPABLE' if latency_performance and IMPORTS_AVAILABLE else '❌ ISSUES DETECTED'}")
        
        if failed_tests > 0:
            print(f"\n❌ Failed Tests:")
            for result in self.results:
                if not result.passed:
                    print(f"   • {result.name}: {result.error}")
        
        overall_success = passed_tests == total_tests and latency_performance and IMPORTS_AVAILABLE
        
        print(f"\n" + "="*60)
        if overall_success:
            print("🎉 OVERALL: SYSTEM READY FOR PRODUCTION")
        else:
            print("⚠️  OVERALL: SYSTEM NEEDS ATTENTION")
            if not IMPORTS_AVAILABLE:
                print("   CRITICAL: Import issues must be resolved first")
        print("="*60)
        
        return report
    
    def save_report(self, filename):
        """Save test report to file"""
        report = self.generate_final_report()
        
        # Ensure test_results directory exists
        os.makedirs("test_results", exist_ok=True)
        filepath = os.path.join("test_results", filename)
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Test report saved: {filepath}")
        return report

# Easy-to-use test runner
def run_comprehensive_tests():
    """Run all tests and generate report"""
    tester = MeditationSystemTester()
    
    try:
        report = tester.run_all_tests()
        
        # Save report with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        tester.save_report(f"meditation_test_report_{timestamp}.json")
        
        return report
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(tester.test_data_dir, ignore_errors=True)

# Main execution
if __name__ == "__main__":
    # Run comprehensive testing
    print("Starting Comprehensive Meditation System Testing...")
    print(f"Working directory: {os.getcwd()}")
    print(f"Python executable: {sys.executable}")
    print()
    
    report = run_comprehensive_tests()
    
    if report:
        success = report['summary']['success_rate_percent'] == 100.0
        real_time = report['summary']['real_time_capable']
        imports_ok = report.get('imports_available', False)
        
        if success and real_time and imports_ok:
            print("\nALL TESTS PASSED - SYSTEM READY FOR MEDITATION SESSIONS!")
        else:
            print("\nSOME TESTS FAILED - REVIEW RESULTS BEFORE PRODUCTION USE")
            
            if not imports_ok:
                print("CRITICAL: Import issues detected - check file locations")
            if not real_time:
                print("PERFORMANCE: System does not meet real-time requirements")
    else:
        print("\nTESTING FAILED - SYSTEM NOT READY")
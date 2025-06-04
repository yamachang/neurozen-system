# tests/run_tests.py

import os
import sys
import subprocess
import time
import json
from datetime import datetime

def print_header():
    """Print test header"""
    print("MEDITATION SYSTEM TEST RUNNER")
    print("=" * 50)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

def check_dependencies():
    """Check if required dependencies are installed"""
    print("Checking dependencies...")
    
    required = ['numpy', 'pandas', 'scipy']
    missing = []
    
    for package in required:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\n Install missing packages:")
        print(f"   pip install {' '.join(missing)}")
        return False
    
    print("✅ All dependencies available")
    return True

def run_quick_test():
    """Run quick smoke test"""
    print("\n Running Quick Smoke Test...")
    
    try:
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        
        from feature_alignment_fix import FeatureAligner
        
        # Quick feature alignment test
        print("   Testing feature alignment...")
        start_time = time.perf_counter()
        
        aligner = FeatureAligner()
        test_features = {'test': 1.0}
        aligned = aligner.align_features(test_features)
        
        duration = (time.perf_counter() - start_time) * 1000
        
        if len(aligned) == 87:
            print(f"   ✅ Feature alignment: {duration:.1f}ms")
            return True
        else:
            print(f"   ❌ Feature alignment failed: got {len(aligned)} features, expected 87")
            return False
            
    except ImportError as e:
        print(f"   ❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return False

def run_full_tests():
    """Run comprehensive test suite"""
    print("\n Running Comprehensive Test Suite...")
    
    try:
        # Run tests
        sys.path.append('.')
        from test_modules import run_comprehensive_tests
        
        print("   This may take 1-2 minutes...")
        report = run_comprehensive_tests()
        
        if report:
            success_rate = report['summary']['success_rate_percent']
            real_time = report['summary']['real_time_capable']
            
            print(f"\n RESULTS:")
            print(f"   Success Rate: {success_rate:.1f}%")
            print(f"   Real-time Capable: {'✅ YES' if real_time else '❌ NO'}")
            
            return success_rate >= 80.0 and real_time
        else:
            print("   ❌ Test suite failed to run")
            return False
            
    except Exception as e:
        print(f"   ❌ Comprehensive tests failed: {e}")
        return False

def run_latency_only():
    """Run only latency tests"""
    print("\n⚡ Running Latency Tests Only...")
    
    try:
        sys.path.append('.')
        from test_comprehensive import MeditationSystemTester
        
        tester = MeditationSystemTester()
        tester.run_latency_tests()
        
        # Check if latency tests passed
        latency_results = [r for r in tester.results if "Latency Test" in r.name]
        all_passed = all(r.passed for r in latency_results)
        
        print(f"\n Latency Results: {'✅ ALL PASSED' if all_passed else '❌ SOME FAILED'}")
        
        for result in latency_results:
            status = "✅" if result.passed else "❌"
            print(f"   {status} {result.name}: {result.duration_ms:.1f}ms")
        
        return all_passed
        
    except Exception as e:
        print(f"   ❌ Latency tests failed: {e}")
        return False

def main():
    """Main test runner"""
    print_header()
    
    # Check dependencies first
    if not check_dependencies():
        print("\n❌ FAILED: Missing dependencies")
        return False
    
    # Get test mode from user
    print("\n Test Options:")
    print("1. Quick Test (30 seconds) - Basic functionality")
    print("2. Latency Only (1 minute) - Performance validation")  
    print("3. Full Test Suite (2-3 minutes) - Complete validation")
    print("4. Exit")
    
    while True:
        try:
            choice = input("\nSelect option (1-4): ").strip()
            
            if choice == '1':
                success = run_quick_test()
                break
            elif choice == '2':
                success = run_latency_only()
                break
            elif choice == '3':
                success = run_full_tests()
                break
            elif choice == '4':
                print("Exiting...")
                return True
            else:
                print("Please enter 1, 2, 3, or 4")
                continue
                
        except KeyboardInterrupt:
            print("\nExiting...")
            return True
        except Exception as e:
            print(f"Error: {e}")
            return False
    
    # Final result
    print("\n" + "="*50)
    if success:
        print("TESTS PASSED - SYSTEM READY")
        print("\n✅ You can now run meditation sessions with confidence:")
        print("   python realtime_stream_web_viz.py")
    else:
        print("⚠️ TESTS FAILED - SYSTEM NEEDS ATTENTION")
        print("\n Recommended actions:")
        print("   1. Check error messages above")
        print("   2. Verify all files are in correct locations")
        print("   3. Ensure LSTM model files exist in ./models/")
        print("   4. Test with simpler configuration first")
    
    print("="*50)
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
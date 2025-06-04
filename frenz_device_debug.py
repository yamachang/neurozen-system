# frenz_device_debug.py - Debug FRENZ device connection and data flow

from frenztoolkit import Streamer
import time
import numpy as np

def debug_frenz_device(device_id, product_key):
    """Debug FRENZ device connection and data flow"""
    print("🔍 FRENZ DEVICE DIAGNOSTIC")
    print("=" * 40)
    print(f"Device: {device_id}")
    print(f"Product Key: {product_key[:15]}...")
    print()
    
    try:
        # Initialize streamer
        print("1️⃣ Initializing FRENZ Streamer...")
        streamer = Streamer(
            device_id=device_id,
            product_key=product_key,
            data_folder="./debug_data",
            turn_off_light=True,
        )
        print("✅ Streamer initialized successfully")
        
        # Start streaming
        print("\n2️⃣ Starting data stream...")
        streamer.start()
        print("✅ Stream started")
        
        # Wait for connection
        print("\n3️⃣ Waiting for device connection (10 seconds)...")
        time.sleep(10)
        
        # Check data for 30 seconds
        print("\n4️⃣ Checking data flow for 30 seconds...")
        
        for i in range(6):  # Check every 5 seconds for 30 seconds
            print(f"\n--- Check #{i+1} (Time: {streamer.session_dur:.1f}s) ---")
            
            # Get raw data
            eeg = streamer.DATA["RAW"]["EEG"]
            imu = streamer.DATA["RAW"]["IMU"]
            
            print(f"EEG Data: {eeg.shape if eeg is not None else 'None'}")
            print(f"IMU Data: {imu.shape if imu is not None else 'None'}")
            
            # Check if we have any samples
            if eeg is not None and eeg.size > 0:
                print(f"✅ EEG: {eeg.shape[0]} samples, {eeg.shape[1]} channels")
                print(f"   Sample values: {eeg[-1, :]} (last sample)")
            else:
                print("❌ EEG: No data received")
            
            if imu is not None and imu.size > 0:
                print(f"✅ IMU: {imu.shape[0]} samples, {imu.shape[1]} channels")
                print(f"   Sample values: {imu[-1, :]} (last sample)")
            else:
                print("❌ IMU: No data received")
            
            # Check API scores
            meditation_score = streamer.SCORES.get("focus_score")
            alpha_power = streamer.SCORES.get("alpha")
            theta_power = streamer.SCORES.get("theta")
            
            print(f"Meditation Score: {meditation_score}")
            print(f"Alpha Power: {alpha_power}")
            print(f"Theta Power: {theta_power}")
            
            if alpha_power is not None:
                print(f"✅ API Scores: Working")
            else:
                print("❌ API Scores: No power scores")
            
            # Check SQC
            sqc_scores = streamer.SCORES.get("sqc_scores")
            print(f"SQC Scores: {sqc_scores}")
            
            time.sleep(5)
        
        # Final diagnosis
        print("\n🏁 FINAL DIAGNOSIS:")
        print("=" * 20)
        
        final_eeg = streamer.DATA["RAW"]["EEG"]
        final_imu = streamer.DATA["RAW"]["IMU"]
        
        if final_eeg is not None and final_eeg.size > 100:
            print("✅ EEG Data: GOOD - Device is streaming EEG data")
        else:
            print("❌ EEG Data: PROBLEM - No or insufficient EEG data")
            print("   Possible causes:")
            print("   • Device not properly worn (poor contact)")
            print("   • Device not powered on")
            print("   • Bluetooth connection unstable")
            print("   • Device firmware issue")
        
        if final_imu is not None and final_imu.size > 100:
            print("✅ IMU Data: GOOD - Device is streaming motion data")
        else:
            print("❌ IMU Data: PROBLEM - No or insufficient IMU data")
        
        if streamer.SCORES.get("alpha") is not None:
            print("✅ API Scores: GOOD - Power analysis working")
        else:
            print("❌ API Scores: PROBLEM - No power scores computed")
            print("   This usually means no EEG data is available")
        
        # Stop streaming
        streamer.stop()
        print("\n✅ Diagnostic complete")
        
        return final_eeg is not None and final_eeg.size > 100
        
    except Exception as e:
        print(f"❌ Error during diagnostic: {e}")
        return False

def suggest_fixes():
    """Suggest fixes for common FRENZ device issues"""
    print("\n🔧 TROUBLESHOOTING STEPS:")
    print("=" * 30)
    print()
    print("1️⃣ DEVICE PHYSICAL CHECK:")
    print("   • Ensure FRENZ device is powered ON")
    print("   • Check that it's properly worn (good skin contact)")
    print("   • Make sure sensors are clean and not blocked")
    print("   • Verify the device LED status")
    print()
    print("2️⃣ BLUETOOTH CONNECTION:")
    print("   • Check Bluetooth settings on your computer")
    print("   • Ensure FRENZ device is paired and connected")
    print("   • Try disconnecting and reconnecting Bluetooth")
    print("   • Move closer to the device (within 3 feet)")
    print()
    print("3️⃣ SOFTWARE CHECKS:")
    print("   • Ensure no other apps are using the FRENZ device")
    print("   • Try restarting the Python script")
    print("   • Check that device_id and product_key are correct")
    print()
    print("4️⃣ DEVICE RESET:")
    print("   • Turn FRENZ device OFF and ON again")
    print("   • Wait 10 seconds between power cycles")
    print("   • Re-pair the device if necessary")
    print()

if __name__ == "__main__":
    # Example usage
    device_id = input("Enter your FRENZ device ID (e.g., FRENZJ12): ").strip().upper()
    product_key = input("Enter your product key: ").strip()
    
    if device_id and product_key:
        success = debug_frenz_device(device_id, product_key)
        
        if not success:
            suggest_fixes()
        else:
            print("\n🎉 Device is working correctly!")
            print("   If your main script still shows no data,")
            print("   the issue might be in the processing pipeline.")
    else:
        print("❌ Please provide valid device ID and product key")
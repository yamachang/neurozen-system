# debug_ppg_data.py - Check PPG data quality

import numpy as np
import sys
import os
import time
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frenztoolkit import Streamer

def analyze_ppg_data():
    """Analyze PPG data quality"""
    
    print("="*60)
    print("PPG DATA QUALITY ANALYSIS")
    print("="*60)
    
    # Initialize streamer
    streamer = Streamer(
        device_id="FRENZJ12",
        product_key="3ix300Z6v7KPkNqFPq96r5ha3kmCEE96MV-KuXyeNXg=",
        data_folder="./debug_ppg_" + time.strftime("%Y%m%d_%H%M%S"),
        turn_off_light=True
    )
    
    try:
        streamer.start()
        print("\nCollecting PPG data for 15 seconds...")
        
        # Wait for data
        time.sleep(15)
        
        # Get PPG data
        ppg_data = streamer.DATA.get("RAW", {}).get("PPG")
        
        if ppg_data is None:
            print("ERROR: No PPG data available!")
            return
            
        print(f"\nPPG Data Analysis:")
        print(f"  Shape: {ppg_data.shape}")
        print(f"  Duration: {ppg_data.shape[0] / 25:.1f} seconds (at 25 Hz)")
        
        # Analyze each channel (Green, Red, IR)
        channel_names = ['Green', 'Red', 'Infrared']
        
        for ch_idx in range(min(3, ppg_data.shape[1])):
            channel_data = ppg_data[:, ch_idx]
            
            print(f"\n{channel_names[ch_idx]} Channel:")
            print(f"  Min: {np.min(channel_data):.2f}")
            print(f"  Max: {np.max(channel_data):.2f}")
            print(f"  Mean: {np.mean(channel_data):.2f}")
            print(f"  Std: {np.std(channel_data):.2f}")
            
            # Check for flat signal
            if np.std(channel_data) < 0.01:
                print(f"  ⚠️ WARNING: Signal appears flat (very low variance)")
            
            # Check for saturation
            unique_vals = len(np.unique(channel_data))
            if unique_vals < 10:
                print(f"  ⚠️ WARNING: Only {unique_vals} unique values - possible saturation")
                
        # Plot first 5 seconds of data
        print("\nPlotting first 5 seconds of PPG data...")
        
        plt.figure(figsize=(12, 8))
        
        # Time axis (5 seconds)
        samples_to_plot = min(125, ppg_data.shape[0])  # 5 seconds at 25 Hz
        time_axis = np.arange(samples_to_plot) / 25.0
        
        for ch_idx in range(min(3, ppg_data.shape[1])):
            plt.subplot(3, 1, ch_idx + 1)
            channel_data = ppg_data[:samples_to_plot, ch_idx]
            
            plt.plot(time_axis, channel_data)
            plt.title(f'{channel_names[ch_idx]} Channel')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('ppg_data_quality.png')
        print("Saved plot to ppg_data_quality.png")
        
        # Check if signal looks like heartbeat
        green_channel = ppg_data[:, 0]  # Usually best for heart rate
        
        # Simple autocorrelation check
        from scipy import signal as scipy_signal
        
        # Normalize
        green_norm = (green_channel - np.mean(green_channel)) / np.std(green_channel)
        
        # Find peaks
        peaks, _ = scipy_signal.find_peaks(green_norm, distance=15)  # Min 15 samples between peaks
        
        if len(peaks) > 2:
            # Estimate heart rate
            peak_intervals = np.diff(peaks) / 25.0  # Convert to seconds
            mean_interval = np.mean(peak_intervals)
            estimated_hr = 60.0 / mean_interval
            
            print(f"\n✅ Detected {len(peaks)} peaks")
            print(f"   Estimated heart rate: {estimated_hr:.1f} BPM")
            
            if 40 < estimated_hr < 180:
                print(f"   Heart rate looks reasonable!")
            else:
                print(f"   ⚠️ Heart rate seems unusual")
        else:
            print(f"\n❌ Could not detect heartbeat pattern")
            print(f"   Only found {len(peaks)} peaks")
            
    finally:
        streamer.stop()
        
    print("\n" + "="*60)
    print("PPG TROUBLESHOOTING:")
    print("1. Make sure the FRENZ device is worn properly")
    print("2. Ensure good skin contact for PPG sensor")
    print("3. Try adjusting the device position")
    print("4. Stay still during measurement")
    print("="*60)

if __name__ == "__main__":
    analyze_ppg_data()
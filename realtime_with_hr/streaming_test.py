# Quick test script
from frenztoolkit import Streamer
import numpy as np
import time

streamer = Streamer(device_id="FRENZJ12", product_key="3ix300Z6v7KPkNqFPq96r5ha3kmCEE96MV-KuXyeNXg=")
streamer.start()
time.sleep(5)  # Let some data accumulate

# Check data format
eeg = streamer.DATA["RAW"]["EEG"]
print(f"EEG type: {type(eeg)}")
print(f"EEG shape: {eeg.shape if hasattr(eeg, 'shape') else 'No shape'}")
if hasattr(eeg, 'shape') and eeg.size > 0:
    print(f"EEG sample (first 5x5):\n{eeg[:5, :5] if eeg.ndim == 2 else eeg[:5]}")

streamer.stop()
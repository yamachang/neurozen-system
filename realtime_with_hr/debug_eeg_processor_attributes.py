# debug_eeg_processor_attributes.py - Find the correct attribute names

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing.eeg_processor import EEGProcessor

# Create an EEG processor instance
processor = EEGProcessor()

print("="*60)
print("EEGProcessor Attributes Related to Quality/Artifacts")
print("="*60)

# Get all attributes
all_attrs = dir(processor)

# Look for attributes related to artifacts, threshold, quality, epoch
relevant_keywords = ['artifact', 'threshold', 'quality', 'epoch', 'gate', 'gating', 'reject', 'percent', 'pct']

print("\nSearching for relevant attributes...")
found_attrs = []

for attr in all_attrs:
    if not attr.startswith('_'):  # Skip private attributes
        for keyword in relevant_keywords:
            if keyword.lower() in attr.lower():
                try:
                    value = getattr(processor, attr)
                    if not callable(value):  # Skip methods
                        found_attrs.append((attr, value))
                        break
                except:
                    pass

print("\nFound attributes:")
for attr_name, attr_value in sorted(set(found_attrs)):
    print(f"  {attr_name}: {attr_value}")

# Also check for any config-related attributes
print("\n\nChecking for config attributes...")
if hasattr(processor, 'config'):
    print("Found 'config' attribute")
    config = processor.config
    if isinstance(config, dict):
        for key, value in config.items():
            if any(kw in str(key).lower() for kw in relevant_keywords):
                print(f"  config['{key}']: {value}")

# Check specific likely names
print("\n\nChecking specific attribute names...")
likely_names = [
    'artifact_threshold',
    'artifact_amplitude_threshold_uv',
    'epoch_artifact_threshold',
    'epoch_quality_threshold',
    'quality_threshold',
    'session_gating',
    'enable_gating',
    'enable_session_gating',
    'enable_session_level_epoch_gating'
]

for name in likely_names:
    if hasattr(processor, name):
        value = getattr(processor, name)
        print(f"✓ Found: {name} = {value}")
    else:
        print(f"✗ Not found: {name}")
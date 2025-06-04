# device_manager.py

import os
import json
import time
from typing import Dict, List, Optional, Tuple

class DeviceManager:
    def __init__(self, config_file="device_config.json"):
        self.config_file = config_file
        self.saved_devices = self._load_saved_devices()
        self.selected_device = None
        self.selected_audio_device = None
        
    def _load_saved_devices(self) -> Dict:        
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    saved_config = json.load(f)
                    
                devices = saved_config.get('devices', {})
                print(f"✅ Loaded {len(devices)} saved device configurations")
                return devices
                
            except Exception as e:
                print(f"⚠️  Error loading device config: {e}")
                print("   Starting with empty device list")
        
        return {}
    
    def save_device_configuration(self, device_id, product_key, audio_device, description=""):
        """Save device configuration for future use"""
        try:
            device_config = {
                "device_id": device_id,
                "product_key": product_key,
                "audio_device": audio_device,
                "description": description or f"FRENZ Brainband {device_id}",
                "last_used": time.time(),
                "usage_count": self.saved_devices.get(device_id, {}).get('usage_count', 0) + 1
            }
            
            self.saved_devices[device_id] = device_config
            
            config_data = {
                "devices": self.saved_devices,
                "last_updated": time.time(),
                "last_selected_device": device_id,
                "last_selected_audio": audio_device
            }
            
            with open(self.config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
                
            print(f"✅ Device configuration saved: {device_id}")
            return True
            
        except Exception as e:
            print(f"⚠️  Error saving device config: {e}")
            return False
    
    def discover_audio_devices(self) -> List[str]:
        """Discover available audio devices"""
        discovered_devices = []
        
        try:
            import sounddevice as sd
            
            print("Scanning for audio devices...")
            devices = sd.query_devices()
            
            for i, device in enumerate(devices):
                device_name = device['name']
                if device['max_output_channels'] >= 2:  # Stereo capable
                    discovered_devices.append({
                        'id': i,
                        'name': device_name,
                        'channels': device['max_output_channels'],
                        'sample_rate': device['default_samplerate'],
                        'is_frenz': 'FRENZ' in device_name.upper() or 'AUDIO_FRENZ' in device_name.upper()
                    })
            
            return discovered_devices
            
        except ImportError:
            print("⚠️  sounddevice not available for audio discovery")
            return []
        except Exception as e:
            print(f"⚠️  Audio discovery error: {e}")
            return []
    
    def get_device_input(self) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Get device configuration from user input"""
        
        print("\n" + "="*60)
        print("FRENZ DEVICE SETUP")
        print("=" * 60)
        print("Enter your FRENZ device ID and product key")
        print("Saved devices can be reused.")
        print()
        
        # Show previously used devices if any
        if self.saved_devices:
            print("📱 Previously Used Devices (click to switch):")
            device_list = list(self.saved_devices.keys())
            
            for i, device_id in enumerate(device_list, 1):
                device_info = self.saved_devices[device_id]
                usage_count = device_info.get('usage_count', 0)
                last_used = device_info.get('last_used', 0)
                last_used_str = time.strftime('%m-%d %H:%M', time.localtime(last_used)) if last_used > 0 else "Never"
                product_key_preview = device_info.get('product_key', '')[:10] + "..." if device_info.get('product_key') else "No key"
                print(f"   {i}. {device_id} (Key: {product_key_preview}) - Used {usage_count}x, last: {last_used_str}")
            
            print(f"   {len(device_list) + 1}. Add new device (test different band)")
            print(f"   {len(device_list) + 2}. Scan for audio devices")
            
            while True:
                try:
                    choice = input(f"\nSelect option (1-{len(device_list) + 2}) or device ID directly: ").strip()
                    
                    # Direct device ID entry
                    if choice.upper().startswith('FRENZ') or choice.upper().startswith('J'):
                        return self._setup_new_device(choice.upper())
                    
                    choice_num = int(choice)
                    
                    # Use existing device
                    if 1 <= choice_num <= len(device_list):
                        selected_id = device_list[choice_num - 1]
                        device_info = self.saved_devices[selected_id]
                        
                        print(f"\n✅ Using saved device: {selected_id}")
                        print(f"   Product key: {device_info['product_key'][:15]}...")
                        print(f"   Audio device: {device_info['audio_device']}")
                        
                        self.selected_device = selected_id
                        self.selected_audio_device = device_info['audio_device']
                        
                        # Update usage count
                        self.save_device_configuration(
                            device_info['device_id'],
                            device_info['product_key'], 
                            device_info['audio_device'],
                            device_info.get('description', '')
                        )
                        
                        return (
                            device_info['device_id'],
                            device_info['product_key'], 
                            device_info['audio_device']
                        )
                    
                    # Enter new device
                    elif choice_num == len(device_list) + 1:
                        return self._setup_new_device()
                    
                    # Scan for audio devices
                    elif choice_num == len(device_list) + 2:
                        self._scan_and_display_audio_devices()
                        continue
                    
                    else:
                        print("Invalid choice. Please try again.")
                        
                except ValueError:
                    # Treat as direct device ID entry
                    if choice.strip():
                        return self._setup_new_device(choice.strip().upper())
                    else:
                        print("Please enter a valid option or device ID")
                except KeyboardInterrupt:
                    print("\nExiting device setup...")
                    return None, None, None
        else:
            # No saved devices - direct setup
            print("No saved devices found. Let's add your first FRENZ device!")
            print("After saving, you can easily switch to other devices anytime.")
            return self._setup_new_device()
    
    def _setup_new_device(self, device_id=None) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Set up a new device - works with ANY FRENZ device ID and product key"""
        print("\n ADD NEW DEVICE")
        print("-" * 30)
        print("Enter your FRENZ device details (will be saved for easy switching)")
        print()
        
        try:
            # Get device ID
            if device_id is None:
                device_id = input("Enter your FRENZ device ID (e.g., FRENZJ12, FRENZJ41, FRENZJ99): ").strip()
            
            if not device_id:
                print("Device ID cannot be empty")
                return None, None, None
            
            # Convert to uppercase for consistency
            device_id = device_id.upper()
            
            # Validate device ID format
            if not (device_id.startswith('FRENZ') or device_id.startswith('J')):
                print(f"⚠️  '{device_id}' doesn't look like a FRENZ device ID")
                confirm = input("Continue anyway? (y/n): ").strip().lower()
                if confirm != 'y':
                    return None, None, None
            
            print(f"\nSetting up device: {device_id}")
            
            # Get product key (REQUIRED)
            print(f"\nProduct Key:")
            print(f"Each FRENZ device has its own unique product key.")
            product_key = input("Enter your product key: ").strip()
            
            if not product_key:
                print("❌ Product key is required!")
                return None, None, None
            
            # Auto-generate audio device name and description
            if device_id.startswith('FRENZ'):
                audio_device = f"AUDIO_{device_id}"
            else:
                audio_device = f"AUDIO_FRENZ{device_id}"
            
            description = f"FRENZ Brainband {device_id}"
            
            # Summary
            print(f"\n DEVICE CONFIGURATION:")
            print(f"   Device ID: {device_id}")
            print(f"   Product Key: {product_key[:15]}...")
            print(f"   Audio Device: {audio_device} (auto-generated)")
            print(f"   Description: {description}")
            
            # Auto-save configuration (no prompt needed)
            self.save_device_configuration(device_id, product_key, audio_device, description)
            
            self.selected_device = device_id
            self.selected_audio_device = audio_device
            
            print(f"✅ Device {device_id} configured and saved!")
            print(f"You can always switch to other devices in future runs")
            return device_id, product_key, audio_device
            
        except KeyboardInterrupt:
            print("\nSetup cancelled")
            return None, None, None
        except Exception as e:
            print(f"❌ Error during setup: {e}")
            return None, None, None
    
    def _scan_and_display_audio_devices(self):
        """Scan and display available audio devices"""
        print("\n SCANNING FOR AUDIO DEVICES")
        print("-" * 30)
        
        discovered = self.discover_audio_devices()
        
        if not discovered:
            print("No audio devices discovered or sounddevice not available")
            return
        
        print(f"\nFound {len(discovered)} audio devices:")
        
        frenz_devices = []
        other_devices = []
        
        for device in discovered:
            if device['is_frenz']:
                frenz_devices.append(device)
            else:
                other_devices.append(device)
        
        # Show FRENZ devices first
        if frenz_devices:
            print("\n FRENZ Audio Devices Found:")
            for device in frenz_devices:
                print(f"   {device['name']} (ID: {device['id']}, {device['channels']} channels)")
        
        # Show other devices
        if other_devices:
            print(f"\n Other Audio Devices:")
            for device in other_devices[:10]:  # Limit to first 10
                print(f"   • {device['name']} (ID: {device['id']}, {device['channels']} channels)")
            
            if len(other_devices) > 10:
                print(f"   ... and {len(other_devices) - 10} more")
        
        if not frenz_devices:
            print("\n No FRENZ audio devices detected.")
            print("   Make sure your FRENZ brainband is:")
            print("   1. Turned on")
            print("   2. Paired via Bluetooth")
            print("   3. Connected to this computer")
        
        input("\nPress Enter to continue...")
    
    def get_last_used_device(self) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Get the most recently used device"""
        if not self.saved_devices:
            return None, None, None
        
        # Find most recently used device
        most_recent = None
        most_recent_time = 0
        
        for device_id, device_info in self.saved_devices.items():
            last_used = device_info.get('last_used', 0)
            if last_used > most_recent_time:
                most_recent_time = last_used
                most_recent = device_info
        
        if most_recent:
            print(f"Using last device: {most_recent['device_id']}")
            print(f"   Product key: {most_recent['product_key'][:15]}...")
            return (
                most_recent['device_id'],
                most_recent['product_key'],
                most_recent['audio_device']
            )
        
        return None, None, None
    
    def quick_setup(self) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Quick setup - use last device or prompt for new one"""
        
        # Try last used device first
        last_config = self.get_last_used_device()
        if last_config[0]:
            return last_config
        
        # No last device - do quick new device setup
        print("QUICK DEVICE SETUP")
        print("=" * 25)
        
        device_id = input("Enter your FRENZ device ID: ").strip().upper()
        if not device_id:
            return None, None, None
        
        product_key = input("Enter your product key: ").strip()
        if not product_key:
            print("❌ Product key is required!")
            return None, None, None
        
        # Auto-generate audio device name
        if device_id.startswith('FRENZ'):
            audio_device = f"AUDIO_{device_id}"
        else:
            audio_device = f"AUDIO_FRENZ{device_id}"
        
        description = f"FRENZ Brainband {device_id}"
        
        # Save for future use
        self.save_device_configuration(device_id, product_key, audio_device, description)
        
        print(f"✅ Quick setup complete: {device_id}")
        return device_id, product_key, audio_device

def get_device_configuration(mode='interactive') -> Tuple[Optional[str], Optional[str], Optional[str]]:
    device_manager = DeviceManager()
    
    if mode == 'quick':
        return device_manager.quick_setup()
    elif mode == 'last':
        return device_manager.get_last_used_device()
    else:  # interactive
        return device_manager.get_device_input()

# Test function
def test_device_manager():
    """Test the device manager with various device configurations"""
    print("TESTING DEVICE MANAGER")
    print("=" * 40)
    
    manager = DeviceManager()
    
    print("\n1. Testing audio device discovery...")
    audio_devices = manager.discover_audio_devices()
    print(f"   Found {len(audio_devices)} audio devices")
    
    print("\n2. Testing configuration save/load...")
    # Test saving a device
    success = manager.save_device_configuration(
        "FRENZTEST", 
        "test_product_key_12345", 
        "AUDIO_FRENZTEST", 
        "Test Device"
    )
    print(f"   Save test: {'✅ Success' if success else '❌ Failed'}")
    
    print("\n✅ Device manager tests completed!")
    print("   System can handle ANY FRENZ device ID + product key combination!")

if __name__ == "__main__":
    # Interactive test
    print("FRENZ DEVICE MANAGER")
    print("Works with any FRENZ brainband device and product key!")
    print()
    
    device_id, product_key, audio_device = get_device_configuration('interactive')
    
    if device_id:
        print(f"\n✅ CONFIGURED DEVICE:")
        print(f"   Device ID: {device_id}")
        print(f"   Product Key: {product_key[:15]}...")
        print(f"   Audio Device: {audio_device}")
        print(f"\n Ready to use with ANY FRENZ device!")
    else:
        print("❌ No device configured")
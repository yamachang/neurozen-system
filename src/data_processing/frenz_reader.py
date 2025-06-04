# src/data_processing/frenz_reader.py

import time
import numpy as np
import os
from datetime import datetime
from frenztoolkit import Streamer # Assuming frenztoolkit is installed
from ..utils.config_manager import load_config

# Load configuration
config = load_config()
eeg_config = config['signal_processing']['eeg']

class FrenzReader:
    """
    Interfaces with the FRENZ brainband to acquire and save biosignal data.
    """
    def __init__(self, device_id=None, product_key=None, data_folder=None):
        self.device_id = device_id or config['frenz_device']['device_id']
        self.product_key = product_key or config['frenz_device']['product_key']
        
        if data_folder is None:
            session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.session_dir = os.path.join(config['paths']['raw_data_dir'], f"session_{session_timestamp}")
        else:
            self.session_dir = data_folder
        os.makedirs(self.session_dir, exist_ok=True)

        self.streamer = Streamer(
            device_id=self.device_id,
            product_key=self.product_key,
            data_folder=self.session_dir # The FRENZ toolkit will save files here
        )
        self.is_streaming = False
        # Cache might not be strictly necessary if toolkit saves continuously and can be read from
        # Or if real-time uses a different mechanism (e.g. streamer callbacks if available)
        self.raw_data_cache = { 
            "EEG": None, "PPG": None, "IMU": None, "FILTERED_EEG": None
        }
        print(f"FrenzReader initialized. FRENZ toolkit will save data to: {self.streamer.data_folder}")
        print(f"Session specific sub-directory (if used by toolkit): {self.session_dir}") # This might be redundant if data_folder is the final path


    def start_streaming(self):
        """Starts the FRENZ data streaming session."""
        if not self.is_streaming:
            try:
                self.streamer.start()
                self.is_streaming = True
                print("FRENZ streaming started.")
                # Allow some data to buffer, useful for initial cache population if needed
                time.sleep(2) 
                # self.update_data_cache() # Populate cache initially if used for real-time
            except Exception as e:
                print(f"Error starting FRENZ streamer: {e}")
                self.is_streaming = False
        else:
            print("Streaming is already active.")

    def stop_streaming(self, save_final_metadata=True):
        """Stops the FRENZ data streaming session. Data is primarily saved by the toolkit."""
        if self.is_streaming:
            print("Stopping FRENZ streaming...")
            # self.update_data_cache(final_fetch=True) # Update cache one last time if used
            
            self.streamer.stop() # This handles the actual saving in frenztoolkit
            self.is_streaming = False
            print(f"FRENZ streaming stopped. Data saved by toolkit to {self.streamer.data_folder}")

            if save_final_metadata:
                self._save_custom_channel_info() # Save our detailed channel info
                # If streamer.stop() doesn't save specific views of data we need (e.g. last N sec), do it here
                # self._save_additional_data_views()
        else:
            print("Streaming is not active.")

    def update_data_cache(self): # Keep if real-time needs a polling mechanism from this cache
        """
        Updates the internal cache with the latest data from the streamer.
        Note: The FRENZ toolkit might offer callbacks or direct buffer access
        that could be more efficient for real-time data acquisition than polling this cache.
        """
        if not self.is_streaming:
            return

        # This assumes streamer.DATA provides access to the current buffers
        if self.streamer.DATA["RAW"]["EEG"] is not None:
            self.raw_data_cache["EEG"] = np.array(self.streamer.DATA["RAW"]["EEG"]).copy()
        if self.streamer.DATA["FILTERED"]["EEG"] is not None:
            self.raw_data_cache["FILTERED_EEG"] = np.array(self.streamer.DATA["FILTERED_EEG"]).copy()
        if self.streamer.DATA["RAW"]["IMU"] is not None:
            self.raw_data_cache["IMU"] = np.array(self.streamer.DATA["RAW"]["IMU"]).copy()
        if self.streamer.DATA["RAW"]["PPG"] is not None:
            self.raw_data_cache["PPG"] = np.array(self.streamer.DATA["RAW"]["PPG"]).copy()
        
    def get_latest_data_segment(self, signal_type="EEG_RAW", n_seconds=None):
        """
        Retrieves the most recent n_seconds of data for a specific signal type from the cache.
        This requires `update_data_cache` to be called regularly.
        This is a simple implementation; a more robust solution would handle asynchronous data streams.
        """
        # self.update_data_cache() # Ensure cache is fresh before access if polled

        data_map = {
            "EEG_RAW": (self.raw_data_cache["EEG"], eeg_config['sample_rate']),
            "EEG_FILTERED": (self.raw_data_cache["FILTERED_EEG"], eeg_config['sample_rate']),
            "PPG_RAW": (self.raw_data_cache["PPG"], config['signal_processing']['ppg']['sample_rate']),
            "IMU_RAW": (self.raw_data_cache["IMU"], config['signal_processing']['imu']['sample_rate']),
        }

        data_array, fs = data_map.get(signal_type, (None, 0))

        if data_array is None or data_array.size == 0:
            return None

        if n_seconds is None:
            return data_array # Return all cached data

        num_samples = int(n_seconds * fs)
        
        if data_array.ndim == 1: # Single channel data
            if len(data_array) >= num_samples:
                return data_array[-num_samples:]
            return data_array
        elif data_array.ndim == 2: # Multi-channel data (channels, samples)
            if data_array.shape[1] >= num_samples:
                return data_array[:, -num_samples:]
            return data_array
        return None


    def _save_custom_channel_info(self):
        """Saves detailed channel information based on config and observations."""
        # The streamer saves its own files. This is for our specific metadata.
        # Path should align with where FRENZ toolkit saves its session data.
        # Assuming streamer.data_folder is the root for session files.
        info_file_path = os.path.join(self.streamer.data_folder, "custom_session_info.txt")
        
        with open(info_file_path, "w") as f:
            f.write(f"Session Timestamp: {os.path.basename(self.streamer.data_folder)}\n") # Assuming folder name is session identifier
            f.write("-" * 30 + "\n")
            f.write("EEG Configuration:\n")
            f.write(f"  Sample Rate: {eeg_config['sample_rate']} Hz\n")
            f.write(f"  Total Channels Provided by Device: {eeg_config['total_channels']}\n")
            f.write(f"  Active Channel Indices Used: {eeg_config['active_channels_indices']}\n")
            f.write(f"  Active Channel Names: {eeg_config['active_channels_names']}\n")
            f.write(f"  Window Size for Features: {eeg_config['feature_window_sec']} s\n")
            f.write(f"  Window Overlap: {eeg_config['feature_overlap_ratio'] * 100}%\n")
            f.write("-" * 30 + "\n")
            
            ppg_conf = config['signal_processing']['ppg']
            f.write("PPG Configuration:\n")
            f.write(f"  Sample Rate: {ppg_conf['sample_rate']} Hz\n")
            f.write(f"  Channels: (Assumed based on common FRENZ, e.g., Green, Red, IR - verify actual count and use)\n")
            f.write("-" * 30 + "\n")

            imu_conf = config['signal_processing']['imu']
            f.write("IMU Configuration:\n")
            f.write(f"  Sample Rate: {imu_conf['sample_rate']} Hz\n")
            f.write(f"  Channels: (Assumed 6: AccX,Y,Z, GyroX,Y,Z - verify actual count and use)\n")
        print(f"Custom session info saved to {info_file_path}")

    def get_session_duration(self):
        """Returns the current session duration in seconds from the streamer."""
        if self.is_streaming and hasattr(self.streamer, 'session_dur'):
            return self.streamer.session_dur
        return 0

    @staticmethod
    def load_session_data(session_data_folder_path):
        """
        Loads all primary .npy data files saved by the FRENZ toolkit from a given session directory.
        The FRENZ toolkit likely names files like `eeg.npy`, `ppg.npy`, `imu.npy`.
        Adjust file names if the toolkit uses different naming conventions.
        """
        data = {}
        # Common names the toolkit might use. Verify from your toolkit's output.
        files_to_load = {
            "raw_eeg": "eeg.npy", # Or raw_eeg.npy if that's what streaming_v1 implies
            "filtered_eeg": "filtered_eeg.npy",
            "raw_ppg": "ppg.npy", # Or raw_ppg.npy
            "raw_imu": "imu.npy"  # Or raw_imu.npy
        }
        
        print(f"Attempting to load data from: {session_data_folder_path}")

        for key, filename in files_to_load.items():
            file_path = os.path.join(session_data_folder_path, filename)
            if os.path.exists(file_path):
                try:
                    # The frenztoolkit saves data; we load it here.
                    # The shape (channels, samples) or (samples, channels) depends on how toolkit saves it.
                    # Your streaming_v1.py suggests (channels, samples) is desired after potential transpose.
                    loaded_array = np.load(file_path, allow_pickle=True)
                    print(f"Loaded {filename} with shape {loaded_array.shape}")
                    
                    # Ensure EEG data matches expected channel count if pre-filtered by toolkit, or handle selection later
                    if key == "raw_eeg" and loaded_array.shape[0] == eeg_config['total_channels']:
                        print(f"  Raw EEG data has {loaded_array.shape[0]} channels as expected.")
                    elif key == "raw_eeg":
                        print(f"  Warning: Raw EEG data has {loaded_array.shape[0]} channels, config expects {eeg_config['total_channels']}. Check consistency.")
                    
                    data[key] = loaded_array
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
            else:
                print(f"File {filename} not found in {session_data_folder_path}")
        
        # Load custom info file if it exists
        custom_info_path = os.path.join(session_data_folder_path, "custom_session_info.txt")
        if os.path.exists(custom_info_path):
            with open(custom_info_path, 'r') as f:
                data['custom_info'] = f.read()
            print(f"Loaded custom_session_info.txt")
            
        return data

if __name__ == '__main__':
    reader = FrenzReader() # Uses paths from config.yaml by default
    
    try:
        print("Starting FRENZ stream for 10 seconds...")
        reader.start_streaming()
        
        start_time = time.time()
        while time.time() - start_time < 10: # Stream for 10 seconds
            time.sleep(1)
            duration = reader.get_session_duration()
            print(f"Session duration: {duration:.2f}s")
            
            # Example: Polling for latest 1s of RAW EEG data (requires update_data_cache to be efficient or called)
            # reader.update_data_cache() 
            # latest_eeg = reader.get_latest_data_segment(signal_type="EEG_RAW", n_seconds=1)
            # if latest_eeg is not None:
            #     print(f"  Latest 1s RAW EEG data from cache (if updated): shape {latest_eeg.shape}")

        print("Streaming finished for this example.")

    except KeyboardInterrupt:
        print("User interrupted.")
    finally:
        print("Stopping stream and saving data (via toolkit and custom info)...")
        reader.stop_streaming(save_final_metadata=True) 
    
    # The actual session directory is determined by the streamer's data_folder parameter
    session_data_path = reader.streamer.data_folder 
    print(f"\nData saved by FRENZ toolkit to: {session_data_path}")

    print("\nAttempting to load data from the session directory using FrenzReader.load_session_data...")
    loaded_session_data = FrenzReader.load_session_data(session_data_path)
    
    if "raw_eeg" in loaded_session_data:
        print(f"Successfully loaded 'raw_eeg' data, shape: {loaded_session_data['raw_eeg'].shape}")
        # Here you could select active channels for further use:
        # active_eeg_data = loaded_session_data['raw_eeg'][eeg_config['active_channels_indices'], :]
        # print(f"Active EEG channels selected, new shape: {active_eeg_data.shape}")
    else:
        print("No 'raw_eeg' data found in loaded_session_data. Check file names output by FRENZ toolkit.")
    
    if 'custom_info' in loaded_session_data:
        print("\nCustom Session Info:\n", loaded_session_data['custom_info'][:300] + "...") # Print first 300 chars
# realtime_stream_web_viz.py - Updated with proper windowing

import numpy as np
import pandas as pd
import json
import time
import asyncio
import websockets
import socket
from datetime import datetime
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import streaming utilities
from streaming_utils import (
    get_streaming_data_with_windowing,
    wait_for_buffer_initialization,
    validate_window_shapes
)

# Import existing modules
from realtime_processor import FixedNormalizationRealTimeProcessor
from lstm_inference import AlignedLSTMInferenceEngine
from device_manager import DeviceManager
from adaptive_audio_system import BulletproofAudioSystem
from participant_manager import ParticipantManager
from frenztoolkit import Streamer

class RealTimeEEGSystem:
    def __init__(self):
        self.device_manager = DeviceManager()
        self.participant_manager = ParticipantManager()
        self.processor = None
        self.inference_engine = None
        self.audio_system = None
        self.streamer = None
        self.websocket = None
        
        # Window configuration
        self.eeg_window_sec = 4.0  # 4-second windows
        self.eeg_window_samples = int(self.eeg_window_sec * 125)  # 500 samples at 125Hz
        self.ppg_window_samples = int(self.eeg_window_sec * 25)   # 100 samples at 25Hz
        self.imu_window_samples = int(self.eeg_window_sec * 50)   # 200 samples at 50Hz
        
        # Processing control
        self.is_running = False
        self.last_process_time = 0
        self.process_interval = 2.0  # Process every 2 seconds
        
    def initialize_components(self):
        """Initialize all system components"""
        print("\n[SYSTEM] Initializing Real-Time EEG System...")
        
        # Initialize processor with normalization
        normalization_path = os.path.join('deployment_assets', 'ml_normalization_stats.json')
        self.processor = FixedNormalizationRealTimeProcessor(
            normalization_stats_path=normalization_path,
            session_buffer_duration_s=30
        )
        
        # Initialize inference engine
        model_dir = 'deployment_assets'
        self.inference_engine = AlignedLSTMInferenceEngine(
            model_path=os.path.join(model_dir, 'lstm_model.h5'),
            scaler_path=os.path.join(model_dir, 'feature_scaler.pkl'),
            feature_columns_path=os.path.join(model_dir, 'feature_columns.txt')
        )
        
        # Initialize audio system
        self.audio_system = BulletproofAudioSystem()
        
        print("[SYSTEM] All components initialized successfully!")
        
    def setup_device_connection(self):
        """Setup FRENZ device connection"""
        device_info = self.device_manager.select_device()
        if not device_info:
            return False
            
        # Initialize streamer
        self.streamer = Streamer(
            device_id=device_info['device_id'],
            product_key=device_info['product_key'],
            data_folder=f"./sessions/{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            turn_off_light=True
        )
        
        return True
        
    def setup_participant_session(self):
        """Setup participant and session"""
        participant_id = self.participant_manager.select_participant()
        session_info = self.participant_manager.select_session(participant_id)
        
        # Configure audio for session
        self.audio_system.set_session_config(
            session_type=session_info['type'],
            session_number=session_info['number']
        )
        
        return participant_id, session_info
        
    async def websocket_handler(self, websocket, path):
        """Handle WebSocket connections"""
        self.websocket = websocket
        print("[WEBSOCKET] Client connected")
        
        try:
            await websocket.wait_closed()
        finally:
            print("[WEBSOCKET] Client disconnected")
            self.websocket = None
            
    async def send_to_client(self, data):
        """Send data to connected WebSocket client"""
        if self.websocket:
            try:
                await self.websocket.send(json.dumps(data))
            except:
                pass
                
    def process_streaming_epoch(self):
        """Process one epoch of streaming data with proper windowing"""
        
        # Get windowed data from streaming buffers
        data = get_streaming_data_with_windowing(
            self.streamer,
            eeg_window_samples=self.eeg_window_samples,
            ppg_window_samples=self.ppg_window_samples,
            imu_window_samples=self.imu_window_samples
        )
        
        # Validate window shapes
        is_valid, error_msg = validate_window_shapes(
            data, 
            expected_eeg_window=self.eeg_window_samples,
            expected_channels=4
        )
        
        if not is_valid:
            print(f"[VALIDATION] Failed: {error_msg}")
            return None
            
        # Process features
        print(f"\n[PROCESSING] Epoch at {self.streamer.session_dur:.1f}s")
        features = self.processor.process_realtime_data(
            eeg_data=data['eeg'],
            ppg_data=data['ppg'],
            imu_data=data['imu'],
            sqc_scores=data['sqc'],
            session_duration=self.streamer.session_dur,
            streamer_scores=self.streamer.SCORES
        )
        
        if not features:
            print("[PROCESSING] No features extracted")
            return None
            
        # Run inference on extracted features
        predictions = []
        for feature_set in features:
            try:
                # Get prediction
                prediction = self.inference_engine.predict(feature_set)
                
                if prediction and 'meditation_state' in prediction:
                    predictions.append(prediction)
                    
                    # Update audio based on prediction
                    self.audio_system.update_meditation_state(
                        prediction['meditation_state'],
                        prediction.get('confidence', 0.5)
                    )
                    
                    # Log results
                    print(f"[PREDICTION] State: {prediction['meditation_state']} "
                          f"(confidence: {prediction.get('confidence', 0):.2f})")
                          
            except Exception as e:
                print(f"[ERROR] Inference failed: {e}")
                
        return {
            'timestamp': time.time(),
            'session_duration': self.streamer.session_dur,
            'features_extracted': len(features),
            'predictions': predictions,
            'buffer_stats': {
                'eeg_samples': data['eeg'].shape[1] if data['eeg'] is not None else 0,
                'ppg_samples': len(data['ppg']) if data['ppg'] is not None else 0,
                'imu_samples': data['imu'].shape[1] if data['imu'] is not None else 0
            }
        }
        
    async def main_processing_loop(self):
        """Main processing loop with proper timing"""
        
        # Start streaming
        self.streamer.start()
        print("\n[STREAMING] Started data collection")
        
        # Wait for buffer initialization
        if not wait_for_buffer_initialization(self.streamer, min_duration_s=10.0):
            print("[ERROR] Buffer initialization failed")
            return
            
        # Start audio
        self.audio_system.start()
        
        # Main processing loop
        self.is_running = True
        epoch_count = 0
        
        try:
            while self.is_running:
                current_time = time.time()
                
                # Process at regular intervals
                if current_time - self.last_process_time >= self.process_interval:
                    epoch_count += 1
                    print(f"\n{'='*60}")
                    print(f"EPOCH {epoch_count} - Duration: {self.streamer.session_dur:.1f}s")
                    print(f"{'='*60}")
                    
                    # Process epoch
                    result = self.process_streaming_epoch()
                    
                    if result:
                        # Send to web client
                        await self.send_to_client({
                            'type': 'epoch_result',
                            'data': result
                        })
                        
                        # Save to file
                        self.save_epoch_result(result)
                        
                    self.last_process_time = current_time
                    
                # Small sleep to prevent CPU overload
                await asyncio.sleep(0.1)
                
                # Check for session end conditions
                if self.streamer.session_dur > 3600:  # 1 hour max
                    print("\n[SESSION] Maximum duration reached")
                    break
                    
        except KeyboardInterrupt:
            print("\n[INTERRUPT] Stopping by user request")
        except Exception as e:
            print(f"\n[ERROR] Processing loop failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
            
    def save_epoch_result(self, result):
        """Save epoch results to file"""
        output_file = os.path.join(
            self.streamer.data_folder,
            'realtime_predictions.jsonl'
        )
        
        with open(output_file, 'a') as f:
            f.write(json.dumps(result) + '\n')
            
    def cleanup(self):
        """Clean up resources"""
        print("\n[CLEANUP] Shutting down system...")
        
        if self.audio_system:
            self.audio_system.stop()
            
        if self.streamer:
            self.streamer.stop()
            
        print("[CLEANUP] Complete")
        
    async def run(self):
        """Main entry point"""
        # Initialize components
        self.initialize_components()
        
        # Setup device
        if not self.setup_device_connection():
            print("[ERROR] Failed to connect to device")
            return
            
        # Setup participant
        participant_id, session_info = self.setup_participant_session()
        print(f"\n[SESSION] Starting {session_info['type']} session {session_info['number']} "
              f"for {participant_id}")
        
        # Start WebSocket server
        ws_server = await websockets.serve(
            self.websocket_handler, 
            'localhost', 
            8765
        )
        print("[WEBSOCKET] Server started on ws://localhost:8765")
        
        # Run main processing loop
        await self.main_processing_loop()
        
        # Cleanup
        ws_server.close()
        await ws_server.wait_closed()
        

# Entry point
if __name__ == "__main__":
    system = RealTimeEEGSystem()
    
    try:
        asyncio.run(system.run())
    except KeyboardInterrupt:
        print("\n[EXIT] Program terminated by user")
    except Exception as e:
        print(f"\n[FATAL] System error: {e}")
        import traceback
        traceback.print_exc()
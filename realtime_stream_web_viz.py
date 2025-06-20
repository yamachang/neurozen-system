# realtime_stream_eeg_imu.py - EEG + IMU ONLY (NO PPG/HR) - SIMPLE HTML VERSION

from frenztoolkit import Streamer
import time
import numpy as np
import os
import json
from datetime import datetime
import copy
import sys

# Import device manager
from device_manager import DeviceManager, get_device_configuration

# Import participant manager
from participant_manager import ParticipantManager

# Import the FIXED normalization processor and LSTM inference (EEG + IMU ONLY)
from realtime_processor import FixedNormalizationRealTimeProcessor
from lstm_inference import AlignedLSTMInferenceEngine

# Import web visualization 
from html_meditation_visualizer import HTMLMeditationVisualizer

# Import FRENZ adaptive audio system
from adaptive_audio_system import MeditationAudioManager

# Safe JSON encoder
class SafeJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

def check_signal_quality_advisory(sqc_scores):
    """Check signal quality as ADVISORY information (not blocking)"""
    if not sqc_scores or len(sqc_scores) == 0:
        return {
            'available': False,
            'good_channels': 0,
            'total_channels': 0,
            'message': 'No SQC data - using software quality assessment',
            'level': 'info'
        }
    
    good_channels = sum(1 for score in sqc_scores if score >= 1)
    total_channels = len(sqc_scores)
    
    if good_channels == total_channels:
        return {
            'available': True,
            'good_channels': good_channels,
            'total_channels': total_channels,
            'message': f'✅ All {total_channels} channels have good signal quality',
            'level': 'excellent'
        }
    elif good_channels >= total_channels // 2:
        return {
            'available': True,
            'good_channels': good_channels,
            'total_channels': total_channels,
            'message': f'✅ {good_channels}/{total_channels} channels have good quality',
            'level': 'good'
        }
    elif good_channels > 0:
        return {
            'available': True,
            'good_channels': good_channels,
            'total_channels': total_channels,
            'message': f'⚠️  Only {good_channels}/{total_channels} good channels (software assessment compensates)',
            'level': 'warning'
        }
    else:
        return {
            'available': True,
            'good_channels': good_channels,
            'total_channels': total_channels,
            'message': f'⚠️  Poor hardware signal quality - relying on software assessment',
            'level': 'poor'
        }

def get_startup_mode():
    """Get startup mode from user"""
    print("\n STARTUP MODE SELECTION")
    print("=" * 40)
    print("1. Interactive Mode - Select saved device or add new one")
    print("2. Quick Start - Use last device OR quick setup for new device")
    print("3. Express Mode - Enter device ID and product key directly (fastest)")
    print()
    print("Easy to switch between different FRENZ devices")
    print()
    
    while True:
        try:
            choice = input("Select mode (1-3): ").strip()
            
            if choice == '1':
                return 'interactive'
            elif choice == '2':
                return 'quick'
            elif choice == '3':
                return 'express'
            else:
                print("Please enter 1, 2, or 3")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            sys.exit(0)

def express_device_setup():
    """Express mode - direct device ID and product key entry"""
    print("\n⚡ EXPRESS DEVICE SETUP")
    print("=" * 25)
    print("Enter your FRENZ device details")
    print()
    
    device_id = input("Enter your FRENZ device ID (e.g., FRENZJ12, FRENZJ41, FRENZJ99): ").strip().upper()
    
    if not device_id:
        print("No device ID entered")
        return None, None, None
    
    product_key = input("Enter your product key: ").strip()
    
    if not product_key:
        print("❌ Product key is required!")
        return None, None, None
    
    # Auto-generate audio device name
    if device_id.startswith('FRENZ'):
        audio_device = f"AUDIO_{device_id}"
    else:
        # Handle cases like "J12" -> "AUDIO_FRENZJ12"
        audio_device = f"AUDIO_FRENZ{device_id}"
    
    print(f"✅ Express setup complete:")
    print(f"   Device: {device_id}")
    print(f"   Product Key: {product_key[:15]}...")
    print(f"   Audio: {audio_device}")
    
    # Save for future use
    try:
        manager = DeviceManager()
        manager.save_device_configuration(
            device_id, 
            product_key, 
            audio_device, 
            f"Express Setup {device_id}"
        )
    except:
        pass  # Don't fail if save doesn't work
    
    return device_id, product_key, audio_device

def main():
    print("FRENZ MEDITATION STUDY SYSTEM - EEG + IMU ONLY")
    print("=" * 55)
    print("PARTICIPANT-BASED SESSION MANAGEMENT (DEID_P# FORMAT)")
    print("   • Structured participant tracking")
    print("   • 6 sessions per participant")
    print("   • Automated data organization")
    print("   • Comprehensive participant reports")
    print("   • DEID_P# naming convention (DEID_P1, DEID_P2, etc.)")
    print("   • 🧠 EEG + 🏃 IMU DATA PROCESSING (NO PPG/HR)")
    print("   • 📊 SIMPLE CLEAN VISUALIZATION")
    print()
    
    # STEP 1: PARTICIPANT MANAGEMENT
    print("STEP 1: PARTICIPANT REGISTRATION")
    print("=" * 40)
    
    # Initialize participant manager
    try:
        participant_manager = ParticipantManager(base_data_dir="./data/DEID_Participants")
        
        # Get participant and session information
        participant_id, session_number = participant_manager.get_participant_input()
        
        if not participant_id or not session_number:
            print("❌ No participant/session selected. Exiting...")
            return
        
        print(f"\n✅ PARTICIPANT SESSION SETUP:")
        print(f"   Participant: {participant_id}")
        print(f"   Session: {session_number}")
        print(f"   Type: {participant_manager.session_config[session_number]['type']}")
        print(f"   Duration: {participant_manager.session_config[session_number]['duration']//60} minutes")
        
        # Create participant session folder
        participant_session_folder = participant_manager.create_participant_session_folder(
            participant_id, session_number
        )
        
    except Exception as e:
        print(f"❌ Error with participant management: {e}")
        return
    
    # STEP 2: DEVICE SELECTION
    print(f"\n" + "="*60)
    print("STEP 2: FRENZ DEVICE SELECTION")
    print("=" * 60)
    print("WORKS WITH ANY FRENZ BRAINBAND DEVICE")
    print("   • Any device ID (FRENZJ12, FRENZJ41, FRENZJ99, etc.)")
    print("   • Any product key")
    print("   • Easy switching between different devices")
    print()
    print("Features:")
    print("• ✅ FLEXIBLE device support (any FRENZ device)")
    print("• ✅ FLEXIBLE product key support (any key)")
    print("• ✅ DIRECT API power scores")
    print("• ✅ SIMPLE, clean visualization (smooth_meditation_visualizer.html)")
    print("• ✅ Dark theme, professional interface")
    print("• ✅ Real-time updates every second")
    print("• ✅ Fixed pre-trained normalization")
    print("• ✅ Software-based quality assessment")
    print("• 🧠 EEG + 🏃 IMU PROCESSING (streamlined, no PPG/HR)")  
    print("• ADAPTIVE BINAURAL BEATS through any FRENZ device")  
    print()
    
    # Get startup mode
    startup_mode = get_startup_mode()
    
    # Flexible device selection
    print(f"\n FLEXIBLE DEVICE SELECTION ({startup_mode.upper()} MODE)")
    print("=" * 55)
    
    if startup_mode == 'interactive':
        device_id, product_key, audio_device = get_device_configuration('interactive')
    elif startup_mode == 'quick':
        device_id, product_key, audio_device = get_device_configuration('quick')
    elif startup_mode == 'express':
        device_id, product_key, audio_device = express_device_setup()
    
    # Check if device selection was successful
    if not device_id or not product_key:
        print("❌ No device configured. Exiting...")
        print("Any FRENZ device ID and product key combination will work")
        return
    
    print(f"\n✅ FLEXIBLE DEVICE CONFIGURATION:")
    print(f"   Device ID: {device_id}")
    print(f"   Product Key: {product_key[:15]}...")
    print(f"   Audio Device: {audio_device or 'None (audio disabled)'}")
    print(f"   System: Ready for ANY FRENZ device + key combination!")
    print()
    
    # STEP 3: SESSION SETUP (Use predetermined session from participant manager)
    print(f"\n" + "="*60)
    print("STEP 3: SESSION CONFIGURATION")
    print("=" * 60)
    
    # Use the session number from participant manager
    selected_session = session_number
    print(f"✅ Session {selected_session} configured for {participant_id}")
    session_info = participant_manager.session_config[selected_session]
    print(f"   Type: {session_info['type']}")
    print(f"   Name: {session_info['name']}")
    print(f"   Duration: {session_info['duration']//60} minutes")
    
    # Initialize FRENZ brainband audio system
    audio_manager = None
    if audio_device:
        print(f"\n🎵 Initializing FRENZ brainband audio system for session {selected_session}...")
        print(f" Target device: {audio_device}")
        try:
            audio_manager = MeditationAudioManager(target_device=audio_device)
            
            if audio_manager.audio_system.audio_device_id is not None:
                print(f"✅ FRENZ brainband audio ready on {audio_device}!")
                print(f"   Device: {device_id}")
                print("   Audio will play directly through your FRENZ device")
                print("   Binaural beats will adapt to your meditation state predictions")
            else:
                print(f"⚠️  Audio device '{audio_device}' not found")
                print(f"   Device: {device_id}")
                print(f"   Check Bluetooth connection for {audio_device}")
                
                use_fallback = input("   Continue with default audio? (y/n): ").lower().strip()
                if use_fallback != 'y':
                    print("Please pair your FRENZ brainband and try again:")
                    print(f"   1. Turn on FRENZ brainband ({device_id})")
                    print(f"   2. System Preferences → Bluetooth → Pair '{audio_device}'")
                    print(f"   3. System Preferences → Sound → Select FRENZ as output")
                    return
                    
                audio_manager = MeditationAudioManager(target_device=None)
                print("⚠️  Using default audio device as fallback")
                
        except Exception as e:
            print(f"⚠️  FRENZ audio system initialization failed: {e}")
            print("   Continuing without audio...")
            audio_manager = None
    else:
        print("No audio device configured - EEG analysis only")
    
    # STEP 4: SYSTEM INITIALIZATION
    print(f"\n" + "="*60)
    print("STEP 4: SYSTEM INITIALIZATION")
    print("=" * 60)
    
    # Initialize the streamer with participant session folder
    print(f"Initializing FRENZ Streamer with device {device_id}...")
    print(f"   This system works with ANY FRENZ device and product key!")
    try:
        # Use participant session folder for streamer data
        streamer = Streamer(
            device_id=device_id,
            product_key=product_key,
            data_folder=participant_session_folder,  # Use participant folder
            turn_off_light=True,
        )
        print(f"✅ Streamer initialized successfully with {device_id}")
        print(f"   Data saving to: {participant_session_folder}")
    except Exception as e:
        print(f"❌ Error initializing streamer with {device_id}: {e}")
        print("   Please check:")
        print(f"   1. Device ID: {device_id} (is this correct?)")
        print(f"   2. Product key: {product_key[:15]}... (is this correct?)")
        print(f"   3. Device is powered on and nearby")
        print(f"   4. No other apps are using the device")
        return

    # Initialize processor with fixed pre-trained normalization
    print("Initializing processor with fixed pre-trained normalization...")
    # FIXED: Check multiple possible locations for normalization stats
    possible_norm_paths = [
        "./data/processed/ml_normalization_stats.json",
        "./ml_normalization_stats.json", 
        "./data/ml_normalization_stats.json",
        "ml_normalization_stats.json"
    ]
    
    normalization_stats_path = None
    for path in possible_norm_paths:
        if os.path.exists(path):
            normalization_stats_path = path
            print(f"✅ Found normalization stats at: {path}")
            break
    
    if normalization_stats_path is None:
        print("⚠️  Normalization stats not found. Checked paths:")
        for path in possible_norm_paths:
            print(f"   - {path} (exists: {os.path.exists(path)})")
        print("   Features will not be normalized")
    
    ml_processor = FixedNormalizationRealTimeProcessor(
        normalization_stats_path=normalization_stats_path,
        session_buffer_duration_s=120
    )

    # Initialize aligned LSTM inference engine
    print("Initializing ALIGNED LSTM inference engine...")
    try:
        lstm_engine = AlignedLSTMInferenceEngine(model_dir="./models")
        lstm_status = lstm_engine.get_status()
        
        if lstm_status['model_loaded']:
            print(f"✅ ALIGNED LSTM model ready!")
            print(f"   Expected features: {lstm_status['expected_features']}")
            print(f"   Sequence length: {lstm_status['sequence_length']}")
        else:
            print("❌ LSTM model failed to load")
            lstm_engine = None
            
    except Exception as e:
        print(f"❌ Error initializing LSTM: {e}")
        lstm_engine = None

    # SIMPLE: Initialize visualization with your existing HTML
    print(f"Initializing SIMPLE meditation visualization for {participant_id}...")
    visualizer = None
    try:
        # Create simple visualizer that works with your existing HTML
        visualizer = HTMLMeditationVisualizer(
            participant_id=participant_id,
            session_number=session_number,
            window_duration=120
        )
        
        if visualizer and visualizer.setup_success:
            # Open your existing smooth_meditation_visualizer.html in browser
            visualizer.show_window()
            print("✅ Simple HTML visualization ready!")
            print("   📄 Using your existing: smooth_meditation_visualizer.html")
            print("   📊 Clean, focused interface with dark theme")
            print("   🔄 Updates every second")
            print(f"   📁 Data file: {visualizer.data_file}")
            print(f"   🌐 Auto-opens at http://localhost:8000/smooth_meditation_visualizer.html")
        else:
            print("⚠️  HTML visualization setup failed")
            visualizer = None
            
    except (ImportError, Exception) as e:
        print(f"⚠️  HTML visualization not available: {e}")
        print("   Continuing without visualization...")
        visualizer = None

    # STEP 5: START SESSION
    print(f"\n" + "="*60)
    print(f"STEP 5: STARTING SESSION")
    print("=" * 60)
    
    # Start streaming
    streamer.start()
    print(f"\n🚀 Started session for {participant_id} (Session {session_number})...")
    print(f"   Device: {device_id}")
    print(f"   Data folder: {participant_session_folder}")
    print(f"   EEG + IMU Processing: ENABLED (no PPG/HR)")
    print(f"   Simple Visualization: ENABLED")
    
    # Connection stabilization
    time.sleep(3)
    
    # START FRENZ AUDIO SESSION
    if audio_manager and selected_session:
        print(f"\n🎵 Starting FRENZ brainband audio session {selected_session}...")
        if audio_manager.initialize_session(selected_session):
            print("✅ FRENZ audio session started!")
            print(f"   Participant: {participant_id}")
            print(f"   Session: {session_number}")
            print("   Binaural beats will adapt to your meditation state")
            print("   Audio is now synchronized with your EEG monitoring")
            
            status = audio_manager.get_status()
            if status.get("active"):
                session_type = status.get("session_type", "unknown")
                duration_min = status.get("total_duration", 0) // 60
                print(f" Session details: {selected_session} ({session_type}, {duration_min} min)")
                if "induction" in session_type.lower():
                    print(f" Binaural beats: ENABLED (adaptive)")
                else:
                    print(f" Binaural beats: DISABLED (sham session)")
        else:
            print("❌ Failed to start FRENZ audio session")
            audio_manager = None
    
    print(f"\n🔬 PROCESSING STARTED:")
    print(f"   Participant: {participant_id}")
    print(f"   Session: {session_number} ({session_info['type']})")
    print(f"   Device: {device_id}")
    print(f"   Data: {participant_session_folder}")
    print(f"   Features: EEG + IMU (no PPG/HR)")
    print(f"   Visualization: Simple, clean interface")
    
    # Data storage with participant-specific filenames (DEID_P# format)
    realtime_features = []
    predictions = []
    
    # Create participant-specific output files with DEID_P# format
    feature_output_file = os.path.join(participant_session_folder, f"ml_features_{participant_id}_S{session_number}_{device_id}.jsonl")
    prediction_output_file = os.path.join(participant_session_folder, f"lstm_predictions_{participant_id}_S{session_number}_{device_id}.jsonl")
    
    print(f"   Features: {os.path.basename(feature_output_file)}")
    print(f"   Predictions: {os.path.basename(prediction_output_file)}")
    
    # MAIN PROCESSING LOOP
    try:
        processing_interval = 4  # Process every 4 seconds
        next_processing_time = time.time() + processing_interval
        session_start_time = time.time()
        
        # Meditation state label mapping
        state_labels = {
            0: 'Rest State',
            1: 'Light Meditation', 
            2: 'Deep Meditation'
        }
        
        while True:
            current_time = time.time()
            
            # Check session duration limits based on selected session
            planned_duration = participant_manager.session_config[session_number]['duration']
            
            if audio_manager and selected_session:
                audio_status = audio_manager.get_status()
                if audio_status.get("active") and audio_status.get("remaining_time", 1) <= 0:
                    print(f"\n✅ FRENZ Session {selected_session} completed for {participant_id}!")
                    break
            else:
                # Check planned duration
                if streamer.session_dur >= planned_duration:
                    print(f"\n✅ Session {session_number} duration reached for {participant_id}!")
                    break

            # Get current data
            eeg = streamer.DATA["RAW"]["EEG"]
            imu = streamer.DATA["RAW"]["IMU"]
            
            # Direct API Power scores
            meditation_score = streamer.SCORES.get("focus_score")
            alpha_power_array = streamer.SCORES.get("alpha")
            theta_power_array = streamer.SCORES.get("theta")
            delta_power_array = streamer.SCORES.get("delta")
            beta_power_array = streamer.SCORES.get("beta")
            gamma_power_array = streamer.SCORES.get("gamma")
            sqc_scores = streamer.SCORES.get("sqc_scores")

            # Add brainwave data to simple visualization (every loop for real-time flow)
            if visualizer and visualizer.setup_success:
                if (alpha_power_array is not None and theta_power_array is not None and 
                    len(alpha_power_array) > 0 and len(theta_power_array) > 0):
                    try:
                        visualizer.add_brainwave_data_from_api(
                            current_time, alpha_power_array, theta_power_array,
                            delta_power_array, beta_power_array, gamma_power_array)
                    except Exception as e:
                        pass

            # Main LSTM processing every 4 seconds
            if current_time >= next_processing_time:
                print(f"\n--- {participant_id} S{session_number} | TIME: {streamer.session_dur:.1f}s | DEVICE: {device_id} ---")
                
                # Show FRENZ audio status
                if audio_manager:
                    audio_status = audio_manager.get_status()
                    if audio_status.get("active"):
                        elapsed_min = audio_status["elapsed_time"] // 60
                        remaining_min = audio_status["remaining_time"] // 60
                        current_freq = audio_status.get("binaural_freq", 0)
                        current_state = audio_status.get("current_state", "Unknown")
                        
                        print(f"FRENZ Audio: Session {audio_status['session_number']} "
                              f"({audio_status['session_type']}) - "
                              f"{elapsed_min:.0f}m elapsed, {remaining_min:.0f}m remaining")
                        print(f"   🎵 Binaural frequency: {current_freq:.1f}Hz (State: {current_state})")
                
                # Show data status with ALL 5 direct API power bands
                print(f"Data: EEG {eeg.shape if eeg is not None else 'None'}, IMU {imu.shape if imu is not None else 'None'}")
                print(f"Meditation Score: {meditation_score}")
                
                print(f"DIRECT API Power (All 5 Bands):")
                if alpha_power_array is not None:
                    print(f"   Alpha: mean {np.mean(alpha_power_array):.1f}dB")
                if theta_power_array is not None:
                    print(f"   Theta: mean {np.mean(theta_power_array):.1f}dB")
                if delta_power_array is not None:
                    print(f"   Delta: mean {np.mean(delta_power_array):.1f}dB")
                if beta_power_array is not None:
                    print(f"   Beta: mean {np.mean(beta_power_array):.1f}dB")
                if gamma_power_array is not None:
                    print(f"   Gamma: mean {np.mean(gamma_power_array):.1f}dB")
                
                # SQC advisory check
                sqc_info = check_signal_quality_advisory(sqc_scores)
                print(f"SQC Advisory: {sqc_info['message']}")
                
                # Process with optimal approach for LSTM (EEG + IMU only)
                if eeg is not None and eeg.size > 0:
                    try:
                        print(f"Processing for LSTM predictions (EEG + IMU only)...")
                        
                        # UPDATED: Pass only EEG and IMU data (no PPG/HR)
                        processed_features = ml_processor.process_realtime_data(
                            eeg_data=eeg,
                            imu_data=imu,
                            sqc_scores=sqc_scores,
                            session_duration=streamer.session_dur,
                            streamer_scores=streamer.SCORES
                        )
                        
                        if processed_features:
                            passed_epochs = sum(1 for f in processed_features if f.get('eeg_quality_flag', False))
                            total_epochs = len(processed_features)
                            print(f"Extracted {total_epochs} epochs ({passed_epochs} passed quality)")
                            
                            # Process each feature epoch for LSTM
                            for epoch_idx, features in enumerate(processed_features):
                                # Add participant metadata to features (DEID_P# format)
                                features['participant_id'] = participant_id
                                features['session_number'] = session_number
                                features['session_type'] = session_info['type']
                                features['realtime_timestamp'] = current_time
                                features['session_duration'] = streamer.session_dur
                                features['device_id'] = device_id
                                features['product_key_preview'] = product_key[:10] + "..."
                                features['audio_device'] = audio_device
                                features['participant_session_folder'] = participant_session_folder
                                features['sqc_advisory_info'] = sqc_info
                                features['streamer_meditation_score'] = meditation_score
                                features['deid_participant_format'] = True
                                features['eeg_imu_processing'] = True  # NEW: Flag for EEG+IMU processing
                                
                                # Store all 5 direct API power bands
                                features['direct_api_alpha_power'] = alpha_power_array.tolist() if alpha_power_array is not None else None
                                features['direct_api_theta_power'] = theta_power_array.tolist() if theta_power_array is not None else None
                                features['direct_api_delta_power'] = delta_power_array.tolist() if delta_power_array is not None else None
                                features['direct_api_beta_power'] = beta_power_array.tolist() if beta_power_array is not None else None
                                features['direct_api_gamma_power'] = gamma_power_array.tolist() if gamma_power_array is not None else None
                                
                                # Add FRENZ audio status
                                if audio_manager:
                                    audio_status = audio_manager.get_status()
                                    features['frenz_audio_status'] = audio_status
                                
                                realtime_features.append(features)
                                
                                # Save features with participant-specific filename
                                with open(feature_output_file, 'a') as f:
                                    f.write(json.dumps(features, cls=SafeJSONEncoder) + '\n')
                                
                                # Extract additional features for visualization
                                if visualizer and visualizer.setup_success:
                                    try:
                                        # Extract the additional features from your processor
                                        additional_features = {
                                            'accel_magnitude_mean': features.get('accel_magnitude_mean', 0.0),
                                            'rf_oter_alpha_plv': features.get('rf_oter_alpha_plv', 0.0),
                                            'rf_oter_theta_plv': features.get('rf_oter_theta_plv', 0.0),
                                            'rf_sef95': features.get('rf_sef95', 20.0),
                                            'oter_sef95': features.get('oter_sef95', 20.0),
                                            'lf_sef95': features.get('lf_sef95', 20.0),
                                            'otel_sef95': features.get('otel_sef95', 20.0)
                                        }
                                        
                                        # Add to visualizer
                                        visualizer.add_additional_features(current_time, additional_features)
                                        print(f"   🔬 Added visualization features: Movement={additional_features['accel_magnitude_mean']:.3f}")
                                    except Exception as e:
                                        print(f"   ⚠️  Additional features error: {e}")                                
                                
                                # LSTM INFERENCE (for quality-passed epochs)
                                if lstm_engine and features.get('eeg_quality_flag', False):
                                    try:
                                        features_for_lstm = copy.deepcopy(features)
                                        lstm_result = lstm_engine.predict(features_for_lstm)
                                        
                                        if lstm_result.get('ready') and lstm_result.get('prediction'):
                                            pred = lstm_result['prediction']
                                            alignment_info = lstm_result.get('alignment_info', {})
                                            
                                            prediction_record = {
                                                'participant_id': participant_id,
                                                'session_number': session_number,
                                                'session_type': session_info['type'],
                                                'timestamp': current_time,
                                                'session_duration': streamer.session_dur,
                                                'epoch_start_time_s': features.get('epoch_start_time_s', 0),
                                                'device_id': device_id,
                                                'product_key_preview': product_key[:10] + "...",
                                                'audio_device': audio_device,
                                                'prediction': pred,
                                                'alignment_info': alignment_info,
                                                'quality_method': features.get('quality_assessment_method', 'unknown'),
                                                'normalization_mode': features.get('normalization_mode', 'unknown'),
                                                'sqc_advisory': sqc_info,
                                                'special_processing': features.get('special_processing', False),
                                                'participant_session_folder': participant_session_folder,
                                                'deid_participant_format': True,
                                                'eeg_imu_processing': True,  # NEW: Flag for EEG+IMU processing
                                                'direct_api_power_available': {
                                                    'alpha': alpha_power_array is not None,
                                                    'theta': theta_power_array is not None,
                                                    'delta': delta_power_array is not None,
                                                    'beta': beta_power_array is not None,
                                                    'gamma': gamma_power_array is not None,
                                                    'alpha_mean': np.mean(alpha_power_array) if alpha_power_array is not None else None,
                                                    'theta_mean': np.mean(theta_power_array) if theta_power_array is not None else None,
                                                    'delta_mean': np.mean(delta_power_array) if delta_power_array is not None else None,
                                                    'beta_mean': np.mean(beta_power_array) if beta_power_array is not None else None,
                                                    'gamma_mean': np.mean(gamma_power_array) if gamma_power_array is not None else None
                                                }
                                            }
                                            
                                            # Add FRENZ audio info to prediction record
                                            if audio_manager:
                                                prediction_record['frenz_audio_status'] = audio_manager.get_status()
                                            
                                            predictions.append(prediction_record)
                                            
                                            # Save prediction with participant-specific filename
                                            with open(prediction_output_file, 'a') as f:
                                                f.write(json.dumps(prediction_record, cls=SafeJSONEncoder) + '\n')
                                            
                                            # Display results with proper state mapping
                                            state_class = pred['class']
                                            state_label = state_labels.get(state_class, f'Unknown State {state_class}')
                                            confidence = pred.get('confidence', 0)
                                            
                                            print(f"LSTM PREDICTION ({participant_id}): {state_label} (Class={state_class}, Confidence={confidence:.3f})")
                                            print(f"   Quality: {features.get('quality_decision_reason', 'unknown')}")
                                            print(f"   Normalization: {features.get('normalization_mode', 'unknown')}")
                                            print(f"   Special: {'Yes' if features.get('special_processing', False) else 'No'}")
                                            print(f"   EEG+IMU Processing: {'Yes' if features.get('eeg_imu_processing', False) else 'No'}")
                                            
                                            # Feature alignment status
                                            if alignment_info.get('alignment_success', False):
                                                print(f"   ✅ Feature alignment: {alignment_info['aligned_feature_count']}/{alignment_info['expected_feature_count']}")
                                            else:
                                                print(f"   ⚠️  Feature alignment: {alignment_info['aligned_feature_count']}/{alignment_info['expected_feature_count']}")
                                            
                                            # Update FRENZ audio system with LSTM prediction
                                            if audio_manager:
                                                try:
                                                    audio_manager.update_from_lstm_prediction(lstm_result)
                                                    print(f"   Updated FRENZ binaural beats: {state_label} → {audio_manager.get_status().get('binaural_freq', 0):.1f}Hz")
                                                except Exception as e:
                                                    print(f"   FRENZ audio update error: {e}")
                                            
                                            # Add meditation prediction to simple visualization
                                            if visualizer and visualizer.setup_success:
                                                try:
                                                    smoothed_class = pred.get('smoothed_class')
                                                    visualizer.add_meditation_prediction(
                                                        current_time, state_class, confidence, smoothed_class)
                                                    print(f"   📊 Added to visualization: {state_label}")
                                                except Exception as e:
                                                    print(f"   ⚠️  Visualization error: {e}")
                                        
                                        elif lstm_result.get('error'):
                                            print(f"LSTM Error: {lstm_result['error']}")
                                    
                                    except Exception as e:
                                        print(f"❌ LSTM error: {e}")
                            
                            # Show processing statistics
                            stats = ml_processor.get_processing_stats()
                            if stats['total_epochs'] > 0:
                                quality_rate = (stats['software_quality_pass'] / stats['total_epochs']) * 100
                                print(f"   Quality Pass Rate: {quality_rate:.1f}% (software-based)")
                                print(f"   Normalization Applied: {'Yes' if stats.get('normalization_applied', 0) > 0 else 'No'}")
                                print(f"   Special Processing: {stats['special_processed_epochs']} epochs")
                                print(f"   EEG+IMU Processing: ENABLED (no PPG/HR)")
                            
                        else:
                            print("Building session context for LSTM...")
                    
                    except Exception as e:
                        print(f"❌ Processing error: {e}")
                        import traceback
                        traceback.print_exc()
                
                next_processing_time = current_time + processing_interval
            
            # Simple Visualization Display (every loop iteration for smooth updates)
            if visualizer and visualizer.setup_success:
                try:
                    visualizer.update_display()  # Simple updates every 0.5 seconds
                except Exception as e:
                    # Silently continue if visualization update fails
                    pass

            time.sleep(0.5)  # 2 Hz main loop for web visualization
            
    except KeyboardInterrupt:
        print(f"\n⏹️  Session stopped by user - {participant_id} Session {session_number}")
    except Exception as e:
        print(f"❌ Error during session: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # STEP 6: SESSION COMPLETION
        print(f"\n" + "="*60)
        print(f"STEP 6: SESSION COMPLETION")
        print("=" * 60)
        
        session_end_time = time.time()
        actual_session_duration = session_end_time - session_start_time
        
        # Cleanup systems
        if audio_manager:
            try:
                audio_manager.stop_current_session()
                print(f"🎵 FRENZ audio session stopped")
            except:
                pass
        
        if visualizer:
            try:
                visualizer.close()
                print("📊 Simple visualization closed")
            except:
                pass
        
        streamer.stop()
        print(f"🔬 EEG streaming stopped")
        
        # Complete session in participant manager
        print(f"\n📋 Completing session for {participant_id}...")
        session_result = participant_manager.complete_session(
            participant_id=participant_id,
            session_number=session_number,
            session_folder=participant_session_folder,
            session_duration_seconds=actual_session_duration,
            device_id=device_id
        )
        
        # Handle session completion result
        if session_result == 'continue':
            print(f"\n🔄 CONTINUING TO NEXT SESSION...")
            print(f"   Please restart the script to begin the next session")
            print(f"   The system will automatically prompt for the next session for {participant_id}")
        elif session_result == True:
            print(f"\n🎉 STUDY COMPLETED FOR {participant_id}!")
        else:
            print(f"\n✅ Session {session_number} completed for {participant_id}")
            print(f"   Run the script again to continue with the next session")
        
        # Session summary
        print(f"\n✅ SESSION SUMMARY:")
        print(f"   Participant: {participant_id}")
        print(f"   Session: {session_number} ({session_info['type']})")
        print(f"   Duration: {actual_session_duration/60:.1f} minutes")
        print(f"   Device: {device_id}")
        print(f"   Features extracted: {len(realtime_features)}")
        print(f"   LSTM predictions: {len(predictions)}")
        print(f"   Data folder: {participant_session_folder}")
        print(f"   EEG+IMU Processing: ENABLED (no PPG/HR)")
        print(f"   Simple Visualization: ENABLED")
        
        # Show session progress and next steps
        completed_count = len(participant_manager.participants[participant_id]['sessions'])
        
        if completed_count >= 6:
            print(f"\n🎉 ALL 6 SESSIONS COMPLETED FOR {participant_id}!")
            print(f"   Comprehensive study report has been generated")
            print(f"   Check ./data/DEID_Participants/reports/ for full analysis")
        else:
            remaining = 6 - completed_count
            print(f"\n📅 Study Progress: {completed_count}/6 sessions completed")
            print(f"   {remaining} sessions remaining for {participant_id}")
            
            next_session_num = participant_manager.get_next_session(
                participant_manager.participants[participant_id]['sessions'].keys()
            )
            if next_session_num:
                next_info = participant_manager.session_config[next_session_num]
                print(f"   Next session: {next_session_num} ({next_info['type']}, {next_info['duration']//60}min)")
                print(f"   Use session management menu to continue or restart sessions")
        
        # Save final device configuration
        if device_id:
            try:
                device_manager = DeviceManager()
                device_manager.save_device_configuration(
                    device_id, product_key, audio_device, f"Session completed {device_id}"
                )
                print(f"✅ Device configuration saved: {device_id}")
            except:
                pass
        
        # Create final session summary (participant-specific)
        if realtime_features:
            final_stats = ml_processor.get_processing_stats()
            
            feature_summary = {
                'participant_id': participant_id,
                'session_number': session_number,
                'session_type': session_info['type'],
                'total_ml_feature_epochs': len(realtime_features),
                'total_lstm_predictions': len(predictions),
                'session_duration': streamer.session_dur,
                'features_per_minute': len(realtime_features) / (streamer.session_dur / 60) if streamer.session_dur > 0 else 0,
                'predictions_per_minute': len(predictions) / (streamer.session_dur / 60) if streamer.session_dur > 0 else 0,
                'participant_session_folder': participant_session_folder,
                'deid_participant_format': True,
                'eeg_imu_processing': True,  # NEW: Flag for EEG+IMU processing
                'simple_visualization': True,  # NEW: Flag for simple visualization
                'device_configuration': {
                    'device_id': device_id,
                    'product_key_preview': product_key[:10] + "...",
                    'audio_device': audio_device,
                    'startup_mode': startup_mode
                },
                'frenz_audio_integration': {
                    'enabled': audio_manager is not None,
                    'device_connected': audio_manager.audio_system.audio_device_id is not None if audio_manager else False,
                    'session_type': audio_manager.get_status().get('session_type') if audio_manager else None,
                    'adaptive_binaural_beats': True,
                    'total_audio_time': streamer.session_dur if audio_manager else 0
                },
                'processing_statistics': final_stats,
                'features_extracted': ['EEG features', 'IMU features'],  # No HR features
                'files_in_session': {
                    'api_raw_data': ['eeg.dat', 'imu.dat', 'spo2.dat'],  # Removed ppg.dat, hr.dat
                    'api_metadata': 'state.json',
                    'api_processed': 'recording_data.npz',
                    'ml_features': f'ml_features_{participant_id}_S{session_number}_{device_id}.jsonl',
                    'lstm_predictions': f'lstm_predictions_{participant_id}_S{session_number}_{device_id}.jsonl',
                    'visualization_data': 'meditation_realtime_data.json'
                }
            }
            
            # Add LSTM statistics
            if lstm_engine and predictions:
                pred_classes = [p['prediction']['class'] for p in predictions]
                unique_classes = list(set(pred_classes))
                class_counts = {cls: pred_classes.count(cls) for cls in unique_classes}
                
                class_confidences = {}
                for cls in unique_classes:
                    cls_preds = [p['prediction']['confidence'] for p in predictions if p['prediction']['class'] == cls]
                    class_confidences[cls] = np.mean(cls_preds) if cls_preds else 0
                
                alignment_successes = [p.get('alignment_info', {}).get('alignment_success', False) for p in predictions]
                alignment_success_rate = sum(alignment_successes) / len(alignment_successes) if alignment_successes else 0
                
                feature_summary['lstm_stats'] = {
                    'total_predictions': len(predictions),
                    'prediction_classes': class_counts,
                    'average_confidence_by_class': class_confidences,
                    'feature_alignment_success_rate': alignment_success_rate,
                    'lstm_status': lstm_engine.get_status(),
                    'state_mapping': {
                        0: 'Rest State',
                        1: 'Light Meditation',
                        2: 'Deep Meditation'
                    }
                }
            
            summary_file = os.path.join(participant_session_folder, f"session_summary_{participant_id}_S{session_number}_{device_id}.json")
            with open(summary_file, 'w') as f:
                json.dump(feature_summary, f, indent=2, cls=SafeJSONEncoder)
            
            print(f"\n🌟 SESSION COMPLETE FOR {participant_id}:")
            print(f" Device Used: {device_id}")
            print(f" Product Key: {product_key[:15]}...")
            if audio_device:
                print(f" Audio Device: {audio_device}")
            print(f" Session folder: {participant_session_folder}")
            print(f" Total feature epochs: {len(realtime_features)}")
            print(f" EEG+IMU Processing: ENABLED (no PPG/HR)")
            print(f" Simple Visualization: ENABLED")
            
            if final_stats['total_epochs'] > 0:
                quality_rate = (final_stats['software_quality_pass'] / final_stats['total_epochs']) * 100
                print(f" Quality Pass Rate: {quality_rate:.1f}% (target: ~95%)")
            
            if lstm_engine and predictions:
                print(f" LSTM predictions: {len(predictions)}")
                pred_classes = [p['prediction']['class'] for p in predictions]
                unique_classes = list(set(pred_classes))
                for cls in unique_classes:
                    count = pred_classes.count(cls)
                    label = state_labels.get(cls, f'Class {cls}')
                    print(f"   {label}: {count} predictions")
            
            print(f"\n SUCCESS - SIMPLE FRENZ MEDITATION SYSTEM:")
            print(f"   ✅ EEG monitoring through FRENZ brainband ({device_id})")
            print(f"   🧠 EEG + 🏃 IMU data processing (streamlined)")
            print(f"   📊 Simple, clean visualization (smooth_meditation_visualizer.html)")
            print(f"   🎨 Dark theme, professional interface")
            print(f"   🔄 Real-time updates every second")
            if audio_device:
                print(f"   ✅ ADAPTIVE audio through FRENZ brainband ({audio_device})")
            print(f"   ✅ Participant data organized in: {participant_session_folder}")

if __name__ == "__main__":
    main()
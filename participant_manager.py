# participant_manager.py

import os
import json
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import shutil
import glob

class ParticipantManager:
    """Manage participants for the meditation study"""
    
    def __init__(self, base_data_dir="./data/DEID_Participants"):
        self.base_data_dir = base_data_dir
        self.participants_config_file = os.path.join(base_data_dir, "participants_registry.json")
        
        # Session configuration (matches adaptive_audio_system.py)
        self.session_config = {
            1: {"duration": 30*60, "type": "INDUCTION", "name": "Complete Meditation Instructions (30min)"},
            2: {"duration": 15*60, "type": "INDUCTION", "name": "Breath Sound Body Meditation (15min)"},
            3: {"duration": 15*60, "type": "SHAM", "name": "Body Scan for Sleep (15min)"},
            4: {"duration": 10*60, "type": "INDUCTION", "name": "Body Scan Guided Meditation (10min)"},
            5: {"duration": 10*60, "type": "INDUCTION", "name": "Guided Imagery Meditation (10min)"},
            6: {"duration": 10*60, "type": "SHAM", "name": "Loving Kindness Meditation (10min)"}
        }
        
        # Ensure directory structure exists
        self.setup_directory_structure()
        
        # Load existing participants
        self.participants = self.load_participants_registry()
        
        print(f"✅ Participant Manager initialized")
        print(f"   Data directory: {self.base_data_dir}")
        print(f"   Registered participants: {len(self.participants)}")
    
    def setup_directory_structure(self):
        """Create necessary directory structure"""
        os.makedirs(self.base_data_dir, exist_ok=True)
        os.makedirs(os.path.join(self.base_data_dir, "reports"), exist_ok=True)
        print(f"Directory structure ready: {self.base_data_dir}")
    
    def load_participants_registry(self):
        """Load existing participants registry"""
        if os.path.exists(self.participants_config_file):
            try:
                with open(self.participants_config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️  Error loading participants registry: {e}")
                return {}
        return {}
    
    def save_participants_registry(self):
        """Save participants registry"""
        try:
            with open(self.participants_config_file, 'w') as f:
                json.dump(self.participants, f, indent=2, default=str)
        except Exception as e:
            print(f"⚠️  Error saving participants registry: {e}")
    
    def validate_participant_id(self, participant_id):
        """Validate participant ID format (DEID_P#)"""
        if not participant_id.startswith('DEID_P'):
            return False
        
        # Extract number part
        try:
            number_part = participant_id[6:]  # After 'DEID_P'
            int(number_part)  # Check if it's a valid number
            return True
        except ValueError:
            return False
    
    def get_participant_input(self):
        """Get participant ID and session selection from user"""
        print("\n" + "="*60)
        print("MEDITATION STUDY - PARTICIPANT REGISTRATION")
        print("=" * 60)
        
        # Show existing participants if any
        if self.participants:
            print(f"Existing participants ({len(self.participants)}):")
            for pid, info in self.participants.items():
                completed_sessions = len(info.get('sessions', {}))
                last_session = max(info.get('sessions', {}).keys()) if info.get('sessions') else 0
                status = "COMPLETE" if completed_sessions == 6 else f"In Progress ({completed_sessions}/6)"
                print(f"   • {pid}: {status}, Last: Session {last_session}")
            print()
        
        # Get participant ID
        while True:
            participant_id = input("Enter Participant ID (e.g., DEID_P1, DEID_P2, DEID_P10): ").strip().upper()
            
            if not participant_id:
                print("❌ Participant ID cannot be empty")
                continue
            
            # Validate DEID_P# format
            if not self.validate_participant_id(participant_id):
                print("❌ Invalid format! Please use: DEID_P1, DEID_P2, DEID_P10, etc.")
                print("   Examples: DEID_P1, DEID_P2, DEID_P10, DEID_P25")
                retry = input("Try again? (y/n): ").lower().strip()
                if retry != 'y':
                    return None, None
                continue
            
            break
        
        # Check if participant exists
        if participant_id in self.participants:
            print(f"\nExisting participant: {participant_id}")
            participant_info = self.participants[participant_id]
            completed_sessions = participant_info.get('sessions', {})
            
            print(f"   Completed sessions: {len(completed_sessions)}/6")
            if completed_sessions:
                for session_num, session_data in completed_sessions.items():
                    session_date = session_data.get('date', 'Unknown')
                    session_type = self.session_config[int(session_num)]['type']
                    print(f"     Session {session_num}: {session_date} ({session_type})")
            else:
                print(f"     No sessions completed yet")
            
            if len(completed_sessions) >= 6:
                print("✅ All sessions completed for this participant!")
            
            # Show session management menu for ALL existing participants
            return self.session_management_menu(participant_id)
            
        else:
            print(f"\nNew participant: {participant_id}")
            # Initialize new participant
            self.participants[participant_id] = {
                'participant_id': participant_id,
                'registration_date': datetime.now().isoformat(),
                'sessions': {},
                'notes': ''
            }
            self.save_participants_registry()
            
            # Even for new participants, show session management menu
            print(f"Participant {participant_id} has been registered.")
            return self.session_management_menu(participant_id)
    
    def session_management_menu(self, participant_id):
        """Show session management menu with options to start next, restart, manually select, or view reports"""
        completed_sessions = list(self.participants[participant_id].get('sessions', {}).keys())
        next_session = self.get_next_session(completed_sessions)
        
        print(f"\nSESSION MANAGEMENT - {participant_id}")
        print("=" * 50)
        
        # Show current status
        print(f"Progress: {len(completed_sessions)}/6 sessions")
        
        if completed_sessions:
            print(f"Completed sessions:")
            for session_num in sorted([int(s) for s in completed_sessions]):
                session_data = self.participants[participant_id]['sessions'][str(session_num)]
                session_type = session_data.get('session_type', 'UNKNOWN')
                session_date = session_data.get('date', 'Unknown')
                print(f"   ✅ Session {session_num}: {session_date} ({session_type})")
        else:
            print(f"No sessions completed yet - ready to start!")
        
        print(f"\nOptions:")
        option_num = 1
        options = {}
        
        # Option 1: Start next session automatically (if available)
        next_session = self.get_next_session(completed_sessions)
        if next_session:
            session_info = self.session_config[next_session]
            print(f"   {option_num}. Start Session {next_session} (next in sequence) - {session_info['name']} ({session_info['type']}, {session_info['duration']//60}min)")
            options[str(option_num)] = ('next', next_session)
            option_num += 1
        else:
            # All sessions completed
            print(f"   {option_num}. All sessions completed! (Choose restart or manual selection)")
            options[str(option_num)] = ('all_complete', None)
            option_num += 1
        
        # Option 2: Manual session selection (always available)
        print(f"   {option_num}. Choose specific session manually (1-6)")
        options[str(option_num)] = ('manual_select', None)
        option_num += 1
        
        # Option 3: Restart any completed session (only if there are completed sessions)
        if completed_sessions:
            print(f"   {option_num}. Restart a completed session")
            options[str(option_num)] = ('restart_menu', None)
            option_num += 1
        
        # Option 4: View reports
        print(f"   {option_num}. View progress report")
        options[str(option_num)] = ('report', None)
        option_num += 1
        
        # Option 5: Exit
        print(f"   {option_num}. Exit")
        options[str(option_num)] = ('exit', None)
        
        # Get user choice
        while True:
            try:
                choice = input(f"\nSelect option (1-{option_num}): ").strip()
                
                if choice in options:
                    action, session_num = options[choice]
                    
                    if action == 'next':
                        # Start next session automatically
                        session_info = self.session_config[session_num]
                        print(f"\n📅 Starting Session {session_num} for {participant_id}")
                        print(f"   Type: {session_info['type']}")
                        print(f"   Name: {session_info['name']}")
                        print(f"   Duration: {session_info['duration']//60} minutes")
                        
                        confirm = input(f"\nStart Session {session_num} for {participant_id}? (y/n): ").lower().strip()
                        if confirm == 'y':
                            return participant_id, session_num
                        else:
                            print("Session cancelled")
                            return None, None
                    
                    elif action == 'manual_select':
                        # Manual session selection (NEW)
                        selected_session = self.manual_session_selection_menu(participant_id, completed_sessions)
                        if selected_session:
                            return participant_id, selected_session
                        else:
                            continue  # Back to main menu
                    
                    elif action == 'restart_menu':
                        # Show restart menu
                        restart_session = self.restart_session_menu(participant_id, completed_sessions)
                        if restart_session:
                            return participant_id, restart_session
                        else:
                            continue  # Back to main menu
                    
                    elif action == 'report':
                        # Generate and show report
                        self.generate_participant_report(participant_id)
                        input("\nPress Enter to continue...")
                        continue  # Back to main menu
                    
                    elif action == 'exit':
                        print("Exiting session management")
                        return None, None
                
                else:
                    print(f"Please enter a number between 1-{option_num}")
                    
            except KeyboardInterrupt:
                print("\nExiting...")
                return None, None
            except ValueError:
                print(f"Please enter a valid number")
    
    def manual_session_selection_menu(self, participant_id, completed_sessions):
        """NEW: Allow manual selection of any session (1-6)"""
        print(f"\nMANUAL SESSION SELECTION - {participant_id}")
        print("=" * 45)
        print("Choose any session (1-6) to start:")
        print()
        
        # Show all sessions with their status
        completed_session_nums = [int(s) for s in completed_sessions]
        
        for session_num in range(1, 7):
            session_info = self.session_config[session_num]
            
            if session_num in completed_session_nums:
                status = "COMPLETED"
                action = "RESTART"
            else:
                status = "NOT STARTED"
                action = "START"
            
            print(f"   {session_num}. Session {session_num} - {session_info['name']}")
            print(f"      Type: {session_info['type']}, Duration: {session_info['duration']//60}min, Status: {status}")
            print(f"      Action: {action} this session")
            print()
        
        print(f"   7. Cancel (back to main menu)")
        
        while True:
            try:
                choice = input(f"\nSelect session to start (1-7): ").strip()
                choice_num = int(choice)
                
                if 1 <= choice_num <= 6:
                    selected_session = choice_num
                    session_info = self.session_config[selected_session]
                    
                    # Check if session is already completed
                    if selected_session in completed_session_nums:
                        print(f"\nSession {selected_session} was already completed")
                        print(f"   Choosing this will RESTART the session")
                        print(f"   Previous data will be archived")
                        print(f"\nSession Details:")
                        print(f"   Type: {session_info['type']}")
                        print(f"   Name: {session_info['name']}")
                        print(f"   Duration: {session_info['duration']//60} minutes")
                        
                        confirm = input(f"\nRESTART Session {selected_session} for {participant_id}? (y/n): ").lower().strip()
                        if confirm == 'y':
                            # Remove the session from completed sessions so it can be restarted
                            self.restart_session(participant_id, selected_session)
                            return selected_session
                        else:
                            print("Session restart cancelled")
                            return None
                    else:
                        print(f"\nStarting NEW Session {selected_session} for {participant_id}")
                        print(f"   Type: {session_info['type']}")
                        print(f"   Name: {session_info['name']}")
                        print(f"   Duration: {session_info['duration']//60} minutes")
                        
                        confirm = input(f"\nStart Session {selected_session} for {participant_id}? (y/n): ").lower().strip()
                        if confirm == 'y':
                            return selected_session
                        else:
                            print("Session start cancelled")
                            return None
                
                elif choice_num == 7:
                    # Cancel
                    return None
                else:
                    print(f"Please enter a number between 1-7")
                    
            except ValueError:
                print("Please enter a valid number")
            except KeyboardInterrupt:
                print("\nCancelling session selection...")
                return None
    
    def restart_session_menu(self, participant_id, completed_sessions):
        """Show menu to select which session to restart"""
        print(f"\nRESTART SESSION - {participant_id}")
        print("=" * 40)
        print("Select which session to restart:")
        
        # Show completed sessions
        for i, session_num in enumerate(sorted([int(s) for s in completed_sessions]), 1):
            session_data = self.participants[participant_id]['sessions'][str(session_num)]
            session_info = self.session_config[session_num]
            session_type = session_data.get('session_type', 'UNKNOWN')
            session_date = session_data.get('date', 'Unknown')
            
            print(f"   {i}. Session {session_num}: {session_info['name']} ({session_type}, {session_info['duration']//60}min)")
            print(f"      Last completed: {session_date}")
        
        print(f"   {len(completed_sessions) + 1}. Cancel (back to main menu)")
        
        while True:
            try:
                choice = input(f"\nSelect session to restart (1-{len(completed_sessions) + 1}): ").strip()
                choice_num = int(choice)
                
                if 1 <= choice_num <= len(completed_sessions):
                    # Get selected session number
                    sorted_sessions = sorted([int(s) for s in completed_sessions])
                    selected_session = sorted_sessions[choice_num - 1]
                    
                    # Confirm restart
                    session_info = self.session_config[selected_session]
                    print(f"\nRESTART Session {selected_session} for {participant_id}")
                    print(f"   Type: {session_info['type']}")
                    print(f"   Name: {session_info['name']}")
                    print(f"   Duration: {session_info['duration']//60} minutes")
                    print(f"   ⚠️  This will overwrite the previous Session {selected_session} data!")
                    
                    confirm = input(f"\nConfirm restart Session {selected_session}? (y/n): ").lower().strip()
                    if confirm == 'y':
                        # Mark session as not completed (remove from registry)
                        self.restart_session(participant_id, selected_session)
                        return selected_session
                    else:
                        print("Restart cancelled")
                        return None
                
                elif choice_num == len(completed_sessions) + 1:
                    # Cancel
                    return None
                else:
                    print(f"Please enter a number between 1-{len(completed_sessions) + 1}")
                    
            except ValueError:
                print("Please enter a valid number")
    
    def restart_session(self, participant_id, session_number):
        """Mark a session as not completed so it can be restarted"""
        try:
            if participant_id in self.participants:
                sessions = self.participants[participant_id].get('sessions', {})
                if str(session_number) in sessions:
                    # Remove the session from completed sessions
                    del sessions[str(session_number)]
                    self.save_participants_registry()
                    print(f"✅ Session {session_number} marked for restart")
                    
                    # Optionally, archive the old session folder
                    participant_dir = os.path.join(self.base_data_dir, participant_id)
                    old_session_dir = os.path.join(participant_dir, f"Session_{session_number}")
                    
                    if os.path.exists(old_session_dir):
                        # Archive old session data
                        archive_dir = os.path.join(participant_dir, f"Session_{session_number}_ARCHIVED_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                        try:
                            shutil.move(old_session_dir, archive_dir)
                            print(f"Previous session data archived to: {os.path.basename(archive_dir)}")
                        except Exception as e:
                            print(f"⚠️  Could not archive old session data: {e}")
                    
        except Exception as e:
            print(f"❌ Error restarting session: {e}")

    def get_next_session(self, completed_sessions):
        """Determine the next session number"""
        completed_nums = [int(s) for s in completed_sessions]
        
        for session_num in range(1, 7):  # Sessions 1-6
            if session_num not in completed_nums:
                return session_num
        
        return None  # All sessions completed
    
    def create_participant_session_folder(self, participant_id, session_number):
        """Create session folder for participant"""
        # Create participant directory
        participant_dir = os.path.join(self.base_data_dir, participant_id)
        os.makedirs(participant_dir, exist_ok=True)
        
        # Create session directory
        session_dir = os.path.join(participant_dir, f"Session_{session_number}")
        os.makedirs(session_dir, exist_ok=True)
        
        # Create session metadata
        session_info = self.session_config[session_number]
        metadata = {
            'participant_id': participant_id,
            'session_number': session_number,
            'session_type': session_info['type'],
            'session_name': session_info['name'],
            'planned_duration_minutes': session_info['duration'] // 60,
            'start_time': datetime.now().isoformat(),
            'status': 'STARTED'
        }
        
        metadata_file = os.path.join(session_dir, "session_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Session folder created: {session_dir}")
        return session_dir
    
    def complete_session(self, participant_id, session_number, session_folder, 
                        session_duration_seconds, device_id=None):
        """Mark session as completed and organize data"""
        try:
            print(f"\nCompleting Session {session_number} for {participant_id}...")
            
            # Update session metadata
            session_dir = os.path.join(self.base_data_dir, participant_id, f"Session_{session_number}")
            metadata_file = os.path.join(session_dir, "session_metadata.json")
            
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            else:
                metadata = {}
            
            metadata.update({
                'end_time': datetime.now().isoformat(),
                'actual_duration_seconds': session_duration_seconds,
                'actual_duration_minutes': session_duration_seconds / 60,
                'status': 'COMPLETED',
                'device_id': device_id,
                'original_session_folder': session_folder
            })
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Copy/move data files from streamer folder to participant folder
            if session_folder and os.path.exists(session_folder):
                self.organize_session_data(session_folder, session_dir, participant_id, device_id)
            
            # Update participants registry
            if participant_id not in self.participants:
                self.participants[participant_id] = {'sessions': {}}
            
            self.participants[participant_id]['sessions'][str(session_number)] = {
                'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'duration_minutes': session_duration_seconds / 60,
                'session_type': self.session_config[session_number]['type'],
                'device_id': device_id,
                'folder': session_dir
            }
            
            self.save_participants_registry()
            
            # Check if all sessions completed
            completed_count = len(self.participants[participant_id]['sessions'])
            print(f"✅ Session {session_number} completed for {participant_id}")
            print(f"   Progress: {completed_count}/6 sessions")
            
            if completed_count >= 6:
                print(f"ALL SESSIONS COMPLETED for {participant_id}!")
                self.generate_participant_report(participant_id)
                return True
            else:
                next_session = self.get_next_session(self.participants[participant_id]['sessions'].keys())
                if next_session:
                    next_info = self.session_config[next_session]
                    print(f"   Next available: Session {next_session} ({next_info['type']}, {next_info['duration']//60}min)")
                
                # Show session management menu
                print(f"\nSession completed! What would you like to do next?")
                return self.show_post_session_menu(participant_id)
            
        except Exception as e:
            print(f"❌ Error completing session: {e}")
            return False
    
    def show_post_session_menu(self, participant_id):
        """Show menu after completing a session"""
        completed_sessions = list(self.participants[participant_id].get('sessions', {}).keys())
        next_session = self.get_next_session(completed_sessions)
        
        print(f"Options:")
        options = {}
        option_num = 1
        
        # Option 1: Start next session automatically
        if next_session:
            session_info = self.session_config[next_session]
            print(f"   {option_num}. Start Session {next_session} now ({session_info['type']}, {session_info['duration']//60}min)")
            options[str(option_num)] = ('next', next_session)
            option_num += 1
        
        # Option 2: Choose session manually
        print(f"   {option_num}. Choose specific session manually (1-6)")
        options[str(option_num)] = ('manual', None)
        option_num += 1
        
        # Option 3: View report
        print(f"   {option_num}. View progress report")
        options[str(option_num)] = ('report', None)
        option_num += 1
        
        # Option 4: Exit
        print(f"   {option_num}. Exit (restart script later for next session)")
        options[str(option_num)] = ('exit', None)
        
        while True:
            try:
                choice = input(f"\nSelect option (1-{option_num}): ").strip()
                
                if choice in options:
                    action, session_num = options[choice]
                    
                    if action == 'next':
                        # Start next session immediately
                        session_info = self.session_config[session_num]
                        print(f"\nContinuing to Session {session_num} for {participant_id}")
                        print(f"   Type: {session_info['type']}")
                        print(f"   Name: {session_info['name']}")
                        print(f"   Duration: {session_info['duration']//60} minutes")
                        
                        confirm = input(f"\nStart Session {session_num} immediately? (y/n): ").lower().strip()
                        if confirm == 'y':
                            print(f"✅ Starting Session {session_num}...")
                            return 'continue'  # Signal to continue with next session
                        else:
                            print("Session cancelled. Exiting...")
                            return False
                    
                    elif action == 'manual':
                        # Manual session selection
                        selected_session = self.manual_session_selection_menu(participant_id, completed_sessions)
                        if selected_session:
                            print(f"✅ Starting Session {selected_session}...")
                            return 'continue'  # Signal to continue with selected session
                        else:
                            continue  # Back to post-session menu
                    
                    elif action == 'report':
                        # Show report and exit
                        self.generate_participant_report(participant_id)
                        return False
                    
                    elif action == 'exit':
                        print("Exiting. Run script again for next session.")
                        return False
                
                else:
                    print(f"Please enter a number between 1-{option_num}")
                    
            except KeyboardInterrupt:
                print("\nExiting...")
                return False
            except ValueError:
                print("Please enter a valid number")
    
    def organize_session_data(self, source_folder, target_folder, participant_id, device_id):
        """Organize and copy session data files"""
        try:
            print(f"Organizing data from {source_folder} to {target_folder}")
            
            # Define file patterns to copy
            file_patterns = [
                "*.jsonl",  # ML features and LSTM predictions
                "*.json",   # Session summaries and metadata
                "*.csv",    # Any CSV exports
                "*.npz",    # Numpy data files
                "recording_data.npz",  # Streamer data
                "state.json",  # Streamer metadata
                "*.dat"     # Raw data files
            ]
            
            copied_files = []
            
            for pattern in file_patterns:
                files = glob.glob(os.path.join(source_folder, pattern))
                for file_path in files:
                    filename = os.path.basename(file_path)
                    target_path = os.path.join(target_folder, filename)
                    
                    try:
                        shutil.copy2(file_path, target_path)
                        copied_files.append(filename)
                    except Exception as e:
                        print(f"   ⚠️  Could not copy {filename}: {e}")
            
            # Create a file inventory
            inventory = {
                'source_folder': source_folder,
                'target_folder': target_folder,
                'copy_timestamp': datetime.now().isoformat(),
                'copied_files': copied_files,
                'participant_id': participant_id,
                'device_id': device_id
            }
            
            inventory_file = os.path.join(target_folder, "data_inventory.json")
            with open(inventory_file, 'w') as f:
                json.dump(inventory, f, indent=2)
            
            print(f"   ✅ Copied {len(copied_files)} files")
            print(f"   File inventory saved: data_inventory.json")
            
        except Exception as e:
            print(f"❌ Error organizing session data: {e}")
    
    def generate_participant_report(self, participant_id):
        """Generate comprehensive participant report"""
        try:
            print(f"\nGenerating report for {participant_id}...")
            
            if participant_id not in self.participants:
                print(f"❌ Participant {participant_id} not found")
                return None
            
            participant_info = self.participants[participant_id]
            sessions = participant_info.get('sessions', {})
            
            # Create report data structure
            report_data = {
                'participant_id': participant_id,
                'registration_date': participant_info.get('registration_date'),
                'report_generated': datetime.now().isoformat(),
                'total_sessions': len(sessions),
                'sessions_summary': {},
                'study_metrics': {},
                'file_locations': {}
            }
            
            # Analyze each session
            total_duration = 0
            session_types = {'INDUCTION': 0, 'SHAM': 0}
            
            for session_num, session_data in sessions.items():
                session_number = int(session_num)
                duration_min = session_data.get('duration_minutes', 0)
                session_type = session_data.get('session_type', 'UNKNOWN')
                
                total_duration += duration_min
                session_types[session_type] = session_types.get(session_type, 0) + 1
                
                # Session details
                report_data['sessions_summary'][session_num] = {
                    'date': session_data.get('date'),
                    'type': session_type,
                    'duration_minutes': duration_min,
                    'device_used': session_data.get('device_id'),
                    'folder': session_data.get('folder'),
                    'planned_duration': self.session_config[session_number]['duration'] // 60
                }
                
                # Analyze session data files if available
                session_folder = session_data.get('folder')
                if session_folder and os.path.exists(session_folder):
                    report_data['file_locations'][session_num] = self.analyze_session_files(session_folder)
            
            # Study-level metrics
            report_data['study_metrics'] = {
                'total_study_time_minutes': total_duration,
                'total_study_time_hours': total_duration / 60,
                'induction_sessions': session_types.get('INDUCTION', 0),
                'sham_sessions': session_types.get('SHAM', 0),
                'average_session_duration': total_duration / len(sessions) if sessions else 0,
                'study_completion_rate': len(sessions) / 6 * 100,
                'study_completed': len(sessions) >= 6
            }
            
            # Save report
            reports_dir = os.path.join(self.base_data_dir, "reports")
            os.makedirs(reports_dir, exist_ok=True)
            
            report_filename = f"{participant_id}_study_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            report_path = os.path.join(reports_dir, report_filename)
            
            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=2)
            
            # Generate human-readable summary
            self.print_participant_summary(report_data)
            
            # Create CSV summary for easy analysis
            self.create_csv_summary(participant_id, report_data)
            
            print(f"Full report saved: {report_path}")
            return report_path
            
        except Exception as e:
            print(f"❌ Error generating participant report: {e}")
            return None
    
    def analyze_session_files(self, session_folder):
        """Analyze files in a session folder"""
        file_info = {
            'folder_path': session_folder,
            'files_found': [],
            'data_files': {},
            'file_sizes_mb': {}
        }
        
        try:
            if os.path.exists(session_folder):
                for filename in os.listdir(session_folder):
                    filepath = os.path.join(session_folder, filename)
                    if os.path.isfile(filepath):
                        file_info['files_found'].append(filename)
                        
                        # Get file size
                        size_mb = os.path.getsize(filepath) / (1024 * 1024)
                        file_info['file_sizes_mb'][filename] = round(size_mb, 2)
                        
                        # Categorize data files
                        if filename.endswith('.jsonl'):
                            if 'ml_features' in filename:
                                file_info['data_files']['ml_features'] = filename
                            elif 'lstm_predictions' in filename:
                                file_info['data_files']['lstm_predictions'] = filename
                        elif filename.endswith('.npz'):
                            file_info['data_files']['raw_recording'] = filename
                        elif filename == 'state.json':
                            file_info['data_files']['session_metadata'] = filename
        
        except Exception as e:
            file_info['error'] = str(e)
        
        return file_info
    
    def print_participant_summary(self, report_data):
        """Print human-readable participant summary"""
        print(f"\n" + "="*70)
        print(f"PARTICIPANT STUDY REPORT: {report_data['participant_id']}")
        print("="*70)
        
        metrics = report_data['study_metrics']
        print(f"Study Overview:")
        print(f"   Total Sessions: {metrics['total_study_time_minutes']:.1f} minutes ({metrics['total_study_time_hours']:.1f} hours)")
        print(f"   Completion Rate: {metrics['study_completion_rate']:.1f}%")
        print(f"   Session Types: {metrics['induction_sessions']} Induction, {metrics['sham_sessions']} Sham")
        
        print(f"\nSession Details:")
        for session_num, session in report_data['sessions_summary'].items():
            planned = session['planned_duration']
            actual = session['duration_minutes']
            completion = (actual / planned * 100) if planned > 0 else 0
            
            print(f"   Session {session_num}: {session['date']}")
            print(f"      Type: {session['type']}")
            print(f"      Duration: {actual:.1f}min (planned: {planned}min, {completion:.1f}%)")
            print(f"      Device: {session.get('device_used', 'Unknown')}")
        
        if metrics['study_completed']:
            print(f"\nSTUDY COMPLETED SUCCESSFULLY!")
        else:
            remaining = 6 - len(report_data['sessions_summary'])
            print(f"\nStudy in progress: {remaining} sessions remaining")
        
        print("="*70)
    
    def create_csv_summary(self, participant_id, report_data):
        """Create CSV summary for data analysis"""
        try:
            # Session-level data
            sessions_data = []
            for session_num, session in report_data['sessions_summary'].items():
                sessions_data.append({
                    'participant_id': participant_id,
                    'session_number': int(session_num),
                    'session_date': session['date'],
                    'session_type': session['type'],
                    'planned_duration_min': session['planned_duration'],
                    'actual_duration_min': session['duration_minutes'],
                    'completion_rate': (session['duration_minutes'] / session['planned_duration'] * 100) if session['planned_duration'] > 0 else 0,
                    'device_id': session.get('device_used', ''),
                    'folder_path': session.get('folder', '')
                })
            
            # Save sessions CSV
            sessions_df = pd.DataFrame(sessions_data)
            sessions_csv = os.path.join(self.base_data_dir, "reports", f"{participant_id}_sessions_summary.csv")
            sessions_df.to_csv(sessions_csv, index=False)
            
            print(f"CSV summary saved: {sessions_csv}")
            
        except Exception as e:
            print(f"⚠️  Could not create CSV summary: {e}")
    
    def get_study_overview(self):
        """Get overview of all participants"""
        print(f"\nSTUDY OVERVIEW")
        print("="*50)
        print(f"Total Participants: {len(self.participants)}")
        
        completed_participants = 0
        total_sessions = 0
        
        for pid, info in self.participants.items():
            session_count = len(info.get('sessions', {}))
            total_sessions += session_count
            if session_count >= 6:
                completed_participants += 1
            
            status = "COMPLETE" if session_count >= 6 else f"{session_count}/6"
            print(f"   {pid}: {status}")
        
        print(f"\nCompleted Studies: {completed_participants}/{len(self.participants)}")
        print(f"Total Sessions Conducted: {total_sessions}")
        completion_rate = (completed_participants / len(self.participants) * 100) if self.participants else 0
        print(f"Study Completion Rate: {completion_rate:.1f}%")

def test_participant_manager():
    """Test the participant manager"""
    print("TESTING PARTICIPANT MANAGER")
    print("=" * 40)
    
    # Initialize manager
    manager = ParticipantManager(base_data_dir="./test_participants")
    
    # Test creating a participant
    test_participant = "DEID_P999"
    test_session = 1
    
    # Create session folder
    session_folder = manager.create_participant_session_folder(test_participant, test_session)
    print(f"✅ Test session folder created: {session_folder}")
    
    # Complete session
    manager.complete_session(test_participant, test_session, session_folder, 1800, "FRENZTEST")
    
    # Generate report
    manager.generate_participant_report(test_participant)
    
    print("✅ Participant manager test completed!")

if __name__ == "__main__":
    test_participant_manager()
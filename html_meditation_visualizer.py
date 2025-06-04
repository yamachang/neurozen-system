# html_meditation_visualizer.py - FIXED VERSION for smooth_meditation_visualizer.html

import json
import time
import numpy as np
import os
from collections import deque
import webbrowser
import http.server
import socketserver
import threading

class HTMLMeditationVisualizer:    
    def __init__(self, data_file=None, participant_id=None, session_number=None, window_duration=120):
        """
        FIXED HTML Meditation Visualizer that works with smooth_meditation_visualizer.html
        """
        
        # Always use simple data file name that your HTML expects
        self.data_file = "meditation_realtime_data.json"
        self.participant_id = participant_id
        self.session_number = session_number
        self.window_duration = window_duration
        self.setup_success = True
        
        # Data storage for smooth_meditation_visualizer.html format
        self.data_points = deque(maxlen=120)  # Store full data points
        self.session_start_time = time.time()
        
        # Current state
        self.current_meditation_score = 0.0
        self.current_meditation_state = 0
        self.current_meditation_label = "Initializing..."
        self.current_confidence = 0.0
        self.current_brainwaves = {
            'delta': -45.0,
            'theta': -40.0, 
            'alpha': -35.0,
            'beta': -30.0,
            'gamma': -25.0
        }
        
        print("✅ FIXED HTML Meditation Visualizer initialized")
        print(f"   Data file: {self.data_file}")
        print(f"   Compatible with: smooth_meditation_visualizer.html")
        print(f"   Participant: {participant_id or 'Not specified'}")
        print(f"   Session: {session_number or 'Not specified'}")
    
    def add_meditation_prediction(self, timestamp, state, confidence, smoothed_class=None):
        """Add meditation prediction in format expected by smooth_meditation_visualizer.html"""
        try:
            # Map states to labels
            state_labels = {
                0: 'Rest State',
                1: 'Light Meditation', 
                2: 'Deep Meditation'
            }
            
            # Calculate simple score: 0=25%, 1=50%, 2=75%
            base_scores = {0: 25.0, 1: 50.0, 2: 75.0}
            base_score = base_scores.get(state, 25.0)
            
            # Add confidence variation (±10%)
            confidence_variation = (confidence - 0.5) * 20
            final_score = np.clip(base_score + confidence_variation, 10, 90)
            
            # Update current state
            self.current_meditation_score = final_score
            self.current_meditation_state = state
            self.current_meditation_label = state_labels.get(state, 'Unknown')
            self.current_confidence = confidence
            
            # Create data point in the format expected by smooth_meditation_visualizer.html
            data_point = {
                'timestamp': timestamp,
                'meditation_state': state,
                'confidence': confidence,
                'brainwaves': self.current_brainwaves.copy(),
                'feature_count': '82 (EEG+IMU)',
                'smoothed_class': smoothed_class or state,
                'score': final_score,
                'label': self.current_meditation_label
            }
            
            # Add to data points
            self.data_points.append(data_point)
            
            print(f"   📊 Meditation: {self.current_meditation_label} ({final_score:.1f}%, conf: {confidence:.3f})")
            
        except Exception as e:
            print(f"⚠️  Error adding meditation prediction: {e}")
    
    def add_brainwave_data_from_api(self, timestamp, alpha_power, theta_power, 
                                  delta_power=None, beta_power=None, gamma_power=None):
        """Add brainwave data from API"""
        try:
            # Calculate averages
            alpha_avg = np.mean(alpha_power) if alpha_power is not None else -35.0
            theta_avg = np.mean(theta_power) if theta_power is not None else -40.0
            delta_avg = np.mean(delta_power) if delta_power is not None else -45.0
            beta_avg = np.mean(beta_power) if beta_power is not None else -30.0
            gamma_avg = np.mean(gamma_power) if gamma_power is not None else -25.0
            
            # Update current brainwaves
            self.current_brainwaves = {
                'delta': float(delta_avg),
                'theta': float(theta_avg),
                'alpha': float(alpha_avg),
                'beta': float(beta_avg),
                'gamma': float(gamma_avg)
            }
            
            # Update the most recent data point if it exists
            if self.data_points:
                self.data_points[-1]['brainwaves'] = self.current_brainwaves.copy()
            
        except Exception as e:
            print(f"⚠️  Error adding brainwave data: {e}")
    
    def update_display(self):
        """Update the HTML display with data format expected by smooth_meditation_visualizer.html"""
        try:
            current_time = time.time()
            elapsed_time = current_time - self.session_start_time
            
            # Create data structure that smooth_meditation_visualizer.html expects
            html_data = {
                # Main data points array (what smooth_meditation_visualizer.html looks for)
                'data_points': list(self.data_points),
                
                # Session info
                'session_info': {
                    'elapsed_time': elapsed_time,
                    'participant_id': self.participant_id,
                    'session_number': self.session_number
                },
                
                # Additional metadata for compatibility
                'last_update': current_time,
                'participant_id': self.participant_id,
                'session_number': self.session_number,
                
                # Current state info
                'current_state': {
                    'meditation_score': self.current_meditation_score,
                    'meditation_state': self.current_meditation_state,
                    'meditation_label': self.current_meditation_label,
                    'meditation_confidence': self.current_confidence,
                    'brainwaves': self.current_brainwaves
                }
            }
            
            # Write to JSON file that smooth_meditation_visualizer.html expects
            with open(self.data_file, 'w') as f:
                json.dump(html_data, f, indent=2)
            
            # Simple debug output
            print(f"   🌐 Updated: {self.current_meditation_label} ({self.current_meditation_score:.1f}%) → {self.data_file}")
            
        except Exception as e:
            print(f"⚠️  Error updating HTML display: {e}")
    
    def show_window(self):
        """Open smooth_meditation_visualizer.html in browser"""
        try:
            # Check if your HTML file exists
            html_file = "polished_neurozen_visualizer.html"
            
            if not os.path.exists(html_file):
                print(f"⚠️  HTML file not found: {html_file}")
                print("   Please make sure smooth_meditation_visualizer.html is in the current directory")
                return False
            
            # Start simple HTTP server
            port = 8000
            while True:
                try:
                    handler = http.server.SimpleHTTPRequestHandler
                    httpd = socketserver.TCPServer(("", port), handler)
                    break
                except OSError:
                    port += 1
                    if port > 8010:
                        print("❌ Could not find available port")
                        return False
            
            # Start server in background
            def serve():
                httpd.serve_forever()
            
            server_thread = threading.Thread(target=serve, daemon=True)
            server_thread.start()
            
            time.sleep(1)  # Let server start
            
            # Open in browser
            url = f'http://localhost:{port}/{html_file}'
            webbrowser.open(url)
            
            print(f"🌐 Opened visualization at: {url}")
            print(f"   Using: {html_file}")
            print(f"   Data file: {self.data_file}")
            print(f"   FIXED data format for smooth_meditation_visualizer.html")
            
            # Store server reference
            self.http_server = httpd
            self.server_thread = server_thread
            
            return True
                
        except Exception as e:
            print(f"⚠️  Error starting server: {e}")
            return False
    
    def close(self):
        """Clean up"""
        try:
            if hasattr(self, 'http_server'):
                self.http_server.shutdown()
                print("Server stopped")
            
            participant_info = f" for {self.participant_id}" if self.participant_id else ""
            print(f"FIXED HTML Visualizer closed{participant_info}")
        except:
            pass


# Simple convenience function
def create_simple_visualizer(participant_id=None, session_number=None):
    """Create FIXED visualizer that works with smooth_meditation_visualizer.html"""
    return HTMLMeditationVisualizer(
        participant_id=participant_id,
        session_number=session_number
    )


if __name__ == "__main__":
    # Test the FIXED visualizer
    print("FIXED HTML Meditation Visualizer Test")
    print("=" * 45)
    
    visualizer = create_simple_visualizer("DEID_P1", 1)
    
    # Test with some data
    import time
    
    for i in range(5):
        current_time = time.time()
        
        # Test meditation prediction
        state = i % 3
        confidence = 0.7 + (i % 3) * 0.1
        visualizer.add_meditation_prediction(current_time, state, confidence)
        
        # Test brainwave data
        alpha = np.random.normal(-35, 3, 4)
        theta = np.random.normal(-40, 3, 4)
        delta = np.random.normal(-45, 3, 4)
        beta = np.random.normal(-30, 3, 4)
        gamma = np.random.normal(-25, 3, 4)
        
        visualizer.add_brainwave_data_from_api(
            current_time, alpha, theta, delta, beta, gamma
        )
        
        visualizer.update_display()
        time.sleep(1)
    
    print(f"\n✅ Test complete - check {visualizer.data_file}")
    print("   Data format is now compatible with smooth_meditation_visualizer.html")
    
    if visualizer.show_window():
        print("🌐 FIXED Visualization opened in browser!")
        input("Press Enter to stop...")
        visualizer.close()
# lstm_inference.py

import os
import sys
import numpy as np
import pandas as pd
import pickle
import json
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# Import the feature aligner
from feature_alignment_fix import FeatureAligner

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

class LSTMInferenceEngine:    
    def __init__(self, model_dir="./models"):
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.sequence_length = 1  # Based on our previously trained offline LSTM
        self.prediction_buffer = deque(maxlen=10)  # Keep last 10 predictions for smoothing
        
        print("Initializing LSTM Inference Engine...")
        self._load_model_components()
        
    def _load_model_components(self):
        """Load the trained model, scaler, and feature configuration"""
        
        # File paths
        model_path = os.path.join(self.model_dir, "best_model.h5")
        scaler_path = os.path.join(self.model_dir, "feature_scaler.pkl")
        features_path = os.path.join(self.model_dir, "feature_columns.txt")
        
        # Check if files exist
        missing_files = []
        for name, path in [("Model", model_path), ("Scaler", scaler_path), ("Features", features_path)]:
            if not os.path.exists(path):
                missing_files.append(f"{name}: {path}")
        
        if missing_files:
            print("❌ Missing model files:")
            for file in missing_files:
                print(f"   {file}")
            raise FileNotFoundError("Required model files not found")
        
        # Load model
        try:
            import tensorflow as tf
            self.model = tf.keras.models.load_model(model_path)
            print(f"✅ Loaded TensorFlow model: {model_path}")
        except ImportError:
            print("❌ TensorFlow not available - required for .h5 model loading")
            raise ImportError("TensorFlow required for .h5 model loading")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
        
        # Load scaler - teammate used joblib.dump()
        try:
            # First try joblib (teammate's method)
            try:
                import joblib
                self.scaler = joblib.load(scaler_path)
                print(f"✅ Loaded feature scaler with joblib: {scaler_path}")
            except ImportError:
                print("⚠️  joblib not available, trying pickle...")
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                print(f"✅ Loaded feature scaler with pickle: {scaler_path}")
            
            print(f"   Scaler type: {type(self.scaler).__name__}")
            
        except Exception as e:
            print(f"❌ Error loading scaler: {e}")
            print("Install joblib with 'pip install joblib scikit-learn'")
            raise
        
        # Load feature columns
        try:
            with open(features_path, 'r') as f:
                self.feature_columns = [line.strip() for line in f if line.strip()]
            print(f"✅ Loaded feature columns: {features_path}")
            print(f"   Expected features: {len(self.feature_columns)}")
            
        except Exception as e:
            print(f"❌ Error loading feature columns: {e}")
            raise
        
        # Verify model input shape matches teammate's training approach
        try:
            input_shape = self.model.input_shape
            print(f"Model input shape: {input_shape}")
            
            if len(input_shape) == 3:  # (batch_size, sequence_length, features)
                model_seq_length = input_shape[1]
                expected_features = input_shape[2]
                
                print(f"   Sequence length: {model_seq_length}")
                print(f"   Features per timestep: {expected_features}")
                
                # Teammate used sequence_length = 1
                if model_seq_length == 1:
                    print("✅ Model matches teammate's training (sequence_length=1)")
                    self.sequence_length = 1
                else:
                    print(f"⚠️  Warning: Model has sequence_length={model_seq_length}, expected 1")
                    self.sequence_length = model_seq_length
                
                # Verify feature count matches
                if expected_features != len(self.feature_columns):
                    print(f"⚠️  Warning: Model expects {expected_features} features, but feature_columns.txt has {len(self.feature_columns)}")
                
            else:
                print(f"⚠️  Unexpected model input shape: {input_shape}")
                self.sequence_length = 1  # Default to teammate's approach
                
        except Exception as e:
            print(f"⚠️  Could not determine sequence length from model: {e}")
            self.sequence_length = 1  # Default to teammate's approach
        
        print(f"   LSTM Inference Engine Ready:")
        print(f"   Model: {os.path.basename(model_path)}")
        print(f"   Features: {len(self.feature_columns)}")
        print(f"   Sequence length: {self.sequence_length}")
        print(f"   Ready for real-time inference!")
        
    def _align_features(self, feature_dict):
        try:
            # Create feature vector in the exact order used during training
            feature_vector = []
            missing_features = []
            
            for feature_name in self.feature_columns:
                if feature_name in feature_dict:
                    value = feature_dict[feature_name]
                    # Handle NaN values
                    if pd.isna(value):
                        feature_vector.append(0.0)  # or use mean imputation
                    else:
                        feature_vector.append(float(value))
                else:
                    missing_features.append(feature_name)
                    feature_vector.append(0.0)  # Default value for missing features
            
            if missing_features:
                print(f"⚠️  Missing features (using defaults): {len(missing_features)}")
                if len(missing_features) <= 5:  # Show first 5
                    print(f"   Examples: {missing_features[:5]}")
            
            return np.array(feature_vector, dtype=np.float32)
            
        except Exception as e:
            print(f"❌ Error aligning features: {e}")
            # Return zero vector as fallback
            return np.zeros(len(self.feature_columns), dtype=np.float32)
    
    def _normalize_features(self, feature_vector):
        try:
            # Reshape for scaler
            feature_vector_2d = feature_vector.reshape(1, -1)
            
            # Apply normalization
            normalized_vector = self.scaler.transform(feature_vector_2d)
            
            return normalized_vector.flatten()
            
        except Exception as e:
            print(f"❌ Error normalizing features: {e}")
            return feature_vector  # Return unnormalized as fallback
    
    def predict(self, feature_dict):
        try:
            # Align features with training format
            feature_vector = self._align_features(feature_dict)
            
            normalized_vector = self._normalize_features(feature_vector)
            
            # Reshape for LSTM input: (batch_size=1, sequence_length=1, features)
            model_input = normalized_vector.reshape(1, 1, -1)
            
            # Make prediction
            prediction = self.model.predict(model_input, verbose=0)
            
            # Process multi-class classification output (softmax)
            pred_probs = prediction[0]  # Get probabilities for each class
            pred_class = int(np.argmax(pred_probs))  # Class with highest probability
            pred_confidence = float(pred_probs[pred_class])  # Confidence = max probability
            
            # Create result
            result = {
                'ready': True,
                'prediction': {
                    'class': pred_class,
                    'confidence': pred_confidence,
                    'probabilities': pred_probs.tolist(),
                    'raw_output': prediction.tolist()
                },
                'model_info': {
                    'input_shape': model_input.shape,
                    'sequence_length': self.sequence_length,
                    'n_features': len(self.feature_columns)
                },
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            # Add to prediction buffer for smoothing
            self.prediction_buffer.append(result['prediction'])
            
            # Add smoothed prediction using recent history
            if len(self.prediction_buffer) >= 3:
                recent_preds = list(self.prediction_buffer)[-3:]  # Last 3 predictions
                
                # Smooth probabilities
                smoothed_probs = np.mean([p['probabilities'] for p in recent_preds], axis=0)
                smoothed_class = int(np.argmax(smoothed_probs))
                smoothed_confidence = float(smoothed_probs[smoothed_class])
                
                result['prediction']['smoothed_class'] = smoothed_class
                result['prediction']['smoothed_confidence'] = smoothed_confidence
                result['prediction']['smoothed_probabilities'] = smoothed_probs.tolist()
            
            return result
            
        except Exception as e:
            print(f"❌ Error making prediction: {e}")
            import traceback
            traceback.print_exc()
            return {
                'ready': False,
                'error': str(e),
                'prediction': None
            }
    
    def get_meditation_state_label(self, prediction_class):
        """Convert prediction class to human-readable label"""
        labels = {
            0: "Rest State", 
            1: "Light Meditation",
            2: "Deep Meditation"
        }
        return labels.get(prediction_class, f"Class_{prediction_class}")
    
    def get_status(self):
        """Get current status of the inference engine"""
        return {
            'model_loaded': self.model is not None,
            'scaler_loaded': self.scaler is not None,
            'features_loaded': self.feature_columns is not None,
            'sequence_length': self.sequence_length,
            'expected_features': len(self.feature_columns) if self.feature_columns else 0,
            'prediction_buffer_size': len(self.prediction_buffer),
            'ready_for_prediction': all([
                self.model is not None, 
                self.scaler is not None, 
                self.feature_columns is not None
            ])
        }

class AlignedLSTMInferenceEngine(LSTMInferenceEngine):    
    def __init__(self, model_dir="./models"):
        super().__init__(model_dir)
        self.feature_aligner = FeatureAligner()
        print("✅ Feature aligner initialized for 87-feature alignment")
        print(f"   Expected features: {len(self.feature_aligner.expected_features)}")
    
    def predict(self, raw_features):
        try:
            # Align features to match LSTM expectations (87 features)
            aligned_features = self.feature_aligner.align_features(raw_features)
            
            # Use the parent predict method with aligned features
            result = super().predict(aligned_features)
            
            # Add alignment info to result
            if result.get('ready'):
                result['alignment_info'] = {
                    'raw_feature_count': len(raw_features),
                    'aligned_feature_count': len(aligned_features),
                    'expected_feature_count': len(self.feature_aligner.expected_features),
                    'alignment_success': len(aligned_features) == len(self.feature_aligner.expected_features)
                }
            
            return result
            
        except Exception as e:
            print(f"❌ Error in aligned prediction: {e}")
            import traceback
            traceback.print_exc()
            return {
                'ready': False,
                'error': str(e),
                'prediction': None
            }

# Test function
def test_aligned_inference_engine():
    """Test the aligned inference engine with dummy data"""
    print("Testing Aligned LSTM Inference Engine...")
    
    try:
        # Initialize aligned engine
        engine = AlignedLSTMInferenceEngine()
        
        # Show status
        status = engine.get_status()
        print(f"\n Engine Status:")
        for key, value in status.items():
            print(f"   {key}: {value}")
        print(f"\n🔬 Testing with mismatched real-time features...")
        
        # Create features that don't perfectly match
        mismatched_features = {
            'LF_ABS_ALPHA_POWER': -1.5,  # Uppercase
            'rf_rel_theta_power': -2.1,  # Lowercase  
            'OTEL_SEF95': 15.2,          # Mixed case
            'heart_rate_87': 72.0,       # Correct
            'stillness_score': 0.8,      # Correct
            'movement_count': 3.0,       # Correct
            'special_processing': True,  # Should be excluded
            'eeg_quality_flag': True,    # Should be excluded
            'session_id': 'test_123',    # Should be excluded
            # Missing many features that will need defaults/interpolation
        }
        
        print(f"   Input features: {len(mismatched_features)}")
        print(f"   Sample inputs: {list(mismatched_features.keys())[:5]}")
        
        # Test prediction
        result = engine.predict(mismatched_features)
        
        if result.get('ready') and result.get('prediction'):
            pred = result['prediction']
            alignment_info = result.get('alignment_info', {})
            
            state_label = engine.get_meditation_state_label(pred['class'])
            print(f"\n Prediction SUCCESS:")
            print(f"   State: {state_label}")
            print(f"   Class: {pred['class']}")
            print(f"   Confidence: {pred['confidence']:.3f}")
            
            print(f"\n Alignment Info:")
            for key, value in alignment_info.items():
                print(f"   {key}: {value}")
            
            # Show class probabilities
            if 'probabilities' in pred:
                print(f"\n Class probabilities:")
                for i, prob in enumerate(pred['probabilities']):
                    label = engine.get_meditation_state_label(i)
                    print(f"   {label}: {prob:.3f}")
        else:
            print(f"❌ Prediction failed: {result.get('error', 'Unknown error')}")
        
        print(f"✅ Aligned inference engine test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_aligned_inference_engine()
# lstm_inference.py

import os
import sys
import numpy as np
import pandas as pd
import pickle
import json
from collections import deque
from pathlib import Path
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
        self._create_normalization_files()
        
    def _create_normalization_files(self):
        """Create normalization files from the loaded scaler if they don't exist"""
        
        ml_stats_path = Path('./data/processed/ml_normalization_stats.json')
        fill_stats_path = Path('./global_feature_fill_stats.json')
        
        # Check if files already exist and are in correct format
        files_need_creation = self._check_normalization_files_format(ml_stats_path, fill_stats_path)
        
        if not files_need_creation:
            print("✅ Normalization files already exist in correct format")
            return
        
        print("🔧 Creating normalization files from training scaler in correct format...")
        
        try:
            # Ensure we have the required components
            if not (self.scaler and self.feature_columns):
                print("❌ Cannot create normalization files: missing scaler or feature columns")
                return
            
            # Verify scaler has the required attributes
            if not (hasattr(self.scaler, 'mean_') and hasattr(self.scaler, 'scale_')):
                print("❌ Scaler missing mean_ or scale_ attributes")
                return
            
            if len(self.scaler.mean_) != len(self.feature_columns):
                print(f"❌ Scaler features ({len(self.scaler.mean_)}) don't match feature names ({len(self.feature_columns)})")
                return
            
            # Create normalization stats dictionary for real-time processor
            normalization_stats = {}
            for i, feature_name in enumerate(self.feature_columns):
                normalization_stats[feature_name] = {
                    'mean': float(self.scaler.mean_[i]),
                    'std': float(self.scaler.scale_[i]),  # StandardScaler stores std in scale_
                    'initialized': True
                }
            
            # Create output directories
            ml_stats_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save ML normalization stats in the correct structure for real-time processor
            # The processor expects {'feature_stats': {...}} structure
            ml_stats_structure = {
                'feature_stats': normalization_stats,  
                'metadata': {
                    'total_features': len(normalization_stats),
                    'source': 'training_scaler',
                    'scaler_type': 'StandardScaler',
                    'created_by': 'lstm_inference_engine'
                }
            }
            
            with open(ml_stats_path, 'w') as f:
                json.dump(ml_stats_structure, f, indent=2)
            
            print(f"✅ Created ML normalization stats: {ml_stats_path}")
            
            # Create global feature fill stats for feature aligner
            fill_stats = {}
            for feature_name, stats in normalization_stats.items():
                fill_stats[feature_name] = {
                    'value': stats['mean'],      # Use mean as fill value
                    'dtype': 'numeric',          # Set as numeric type
                    'source': 'training_mean'    # Source information
                }
            
            with open(fill_stats_path, 'w') as f:
                json.dump(fill_stats, f, indent=2)
            
            print(f"✅ Created global fill stats: {fill_stats_path}")
            print(f"   📊 Created stats for {len(normalization_stats)} features")
            
            # Show some example stats
            print(f"   Sample normalization statistics:")
            sample_features = list(normalization_stats.items())[:3]
            for feature_name, stats in sample_features:
                print(f"   {feature_name:30}: mean={stats['mean']:8.3f}, std={stats['std']:8.3f}")
            
            # Verify the files were created correctly
            verification_result = self._verify_normalization_files_format()
            if verification_result:
                print(f"✅ File formats verified - should fix both feature aligner and real-time processor!")
            else:
                print(f"⚠️  File format verification failed")
            
        except Exception as e:
            print(f"❌ Error creating normalization files: {e}")
            import traceback
            traceback.print_exc()
    
    def _check_normalization_files_format(self, ml_stats_path, fill_stats_path):
        """Check if normalization files exist and are in correct format"""
        if not ml_stats_path.exists() or not fill_stats_path.exists():
            return True
        
        try:
            # Check ML stats format
            with open(ml_stats_path, 'r') as f:
                ml_data = json.load(f)
            
            # Should have feature_stats wrapper and features with mean/std
            if not ('feature_stats' in ml_data and 
                    isinstance(ml_data['feature_stats'], dict) and
                    ml_data['feature_stats'] and 
                    all(isinstance(v, dict) and 'mean' in v and 'std' in v 
                        for v in ml_data['feature_stats'].values())):
                print("🔧 ML stats file has incorrect format, will recreate...")
                return True
            
            # Check fill stats format  
            with open(fill_stats_path, 'r') as f:
                fill_data = json.load(f)
            
            # Should have features with value/dtype/source structure
            if not fill_data or not all(isinstance(v, dict) and 'value' in v and 'dtype' in v for v in fill_data.values()):
                print("🔧 Fill stats file has incorrect format, will recreate...")
                return True
                
            # Files exist and have correct format
            return False
            
        except Exception as e:
            print(f"🔧 Error checking file formats: {e}, will recreate...")
            return True
    
    def _verify_normalization_files_format(self):
        """Verify that normalization files were created in correct format"""
        
        try:
            ml_stats_path = Path('./data/processed/ml_normalization_stats.json')
            fill_stats_path = Path('./global_feature_fill_stats.json')
            
            # Check ML stats
            if ml_stats_path.exists():
                with open(ml_stats_path, 'r') as f:
                    ml_data = json.load(f)
                
                feature_stats = ml_data.get('feature_stats', {})
                print(f"   ✅ ML stats format: {len(feature_stats)} features in feature_stats wrapper")
                
                if len(feature_stats) == 0:
                    print(f"   ❌ ML stats has no features!")
                    return False
                    
            else:
                print(f"   ❌ ML stats file missing")
                return False
            
            # Check fill stats
            if fill_stats_path.exists():
                with open(fill_stats_path, 'r') as f:
                    fill_data = json.load(f)
                
                # Verify format
                sample_key = list(fill_data.keys())[0] if fill_data else None
                if sample_key and isinstance(fill_data[sample_key], dict):
                    sample_entry = fill_data[sample_key]
                    if 'value' in sample_entry and 'dtype' in sample_entry:
                        print(f"   ✅ Fill stats format: {len(fill_data)} features with proper metadata")
                        print(f"   Sample: {sample_key} = {sample_entry}")
                    else:
                        print(f"   ❌ Fill stats missing required fields")
                        return False
                else:
                    print(f"   ❌ Fill stats has wrong structure")
                    return False
            else:
                print(f"   ❌ Fill stats file missing")
                return False
            
            return True
            
        except Exception as e:
            print(f"   ❌ Verification error: {e}")
            return False
        
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
            
            # Show scaler stats for verification
            if hasattr(self.scaler, 'mean_') and hasattr(self.scaler, 'scale_'):
                print(f"   Scaler mean range: [{self.scaler.mean_.min():.6f}, {self.scaler.mean_.max():.6f}]")
                print(f"   Scaler scale range: [{self.scaler.scale_.min():.6f}, {self.scaler.scale_.max():.6f}]")
            
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
    
    def verify_normalization_files(self):
        """Verify that normalization files were created successfully"""
        ml_stats_path = Path('./data/processed/ml_normalization_stats.json')
        fill_stats_path = Path('./global_feature_fill_stats.json')
        
        results = {
            'ml_stats_exists': ml_stats_path.exists(),
            'fill_stats_exists': fill_stats_path.exists(),
            'ml_stats_size': 0,
            'fill_stats_size': 0
        }
        
        if results['ml_stats_exists']:
            try:
                with open(ml_stats_path, 'r') as f:
                    ml_data = json.load(f)
                
                # Check for proper structure
                feature_stats = ml_data.get('feature_stats', {})
                results['ml_stats_size'] = len(feature_stats)
                results['ml_stats_has_wrapper'] = 'feature_stats' in ml_data
                
            except:
                results['ml_stats_size'] = -1
        
        if results['fill_stats_exists']:
            try:
                with open(fill_stats_path, 'r') as f:
                    fill_data = json.load(f)
                results['fill_stats_size'] = len(fill_data)
            except:
                results['fill_stats_size'] = -1
        
        return results

class AlignedLSTMInferenceEngine(LSTMInferenceEngine):    
    def __init__(self, model_dir="./models"):
        super().__init__(model_dir)
        self.feature_aligner = FeatureAligner()
        print("✅ Feature aligner initialized for 87-feature alignment")
        print(f"   Expected features: {len(self.feature_aligner.expected_features)}")
        
        # Verify normalization files were created
        norm_check = self.verify_normalization_files()
        if norm_check['ml_stats_exists'] and norm_check['fill_stats_exists']:
            print(f"✅ Normalization files verified:")
            print(f"   ML stats: {norm_check['ml_stats_size']} features")
            print(f"   Fill stats: {norm_check['fill_stats_size']} features")
            print(f"🎯 Real-time processor should now apply proper normalization!")
        else:
            print(f"⚠️  Normalization files missing - real-time normalization may fail")
    
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

# Test function with normalization verification
def test_aligned_inference_engine():
    """Test the aligned inference engine with dummy data"""
    print("Testing Enhanced Aligned LSTM Inference Engine...")
    
    try:
        # Initialize aligned engine
        engine = AlignedLSTMInferenceEngine()
        
        # Show status
        status = engine.get_status()
        print(f"\n📊 Engine Status:")
        for key, value in status.items():
            print(f"   {key}: {value}")
        
        # Verify normalization files
        norm_status = engine.verify_normalization_files()
        print(f"\n📊 Normalization Files Status:")
        for key, value in norm_status.items():
            print(f"   {key}: {value}")
        
        print(f"\n🔬 Testing Feature Aligner Compatibility...")
        
        # Test that feature aligner works (this was failing before)
        try:
            from feature_alignment_fix import FeatureAligner
            test_aligner = FeatureAligner()
            
            simple_features = {
                'lf_abs_theta_power': -35.0,
                'rf_abs_theta_power': -33.0,
                'heart_rate_87': 72.0
            }
            
            aligned = test_aligner.align_features(simple_features)
            print(f"   ✅ Feature aligner working: {len(simple_features)} → {len(aligned)} features")
            
        except Exception as e:
            print(f"   ❌ Feature aligner still failing: {e}")
            return False
        
        print(f"\n🔬 Testing LSTM Predictions with diverse scenarios...")
        
        # Test with scenarios that should produce different predictions
        test_scenarios = [
            {
                'name': 'Low Theta (Rest-like)',
                'features': {
                    'lf_abs_theta_power': -45.0,  # Very low theta
                    'rf_abs_theta_power': -44.0,
                    'lf_rel_theta_power': -3.0,
                    'rf_rel_theta_power': -2.8,
                    'heart_rate_87': 85.0,
                    'stillness_score': 0.3,
                    'movement_count': 8.0
                }
            },
            {
                'name': 'Medium Theta (Light meditation)',
                'features': {
                    'lf_abs_theta_power': -33.0,  # Medium theta
                    'rf_abs_theta_power': -32.0,
                    'lf_rel_theta_power': -1.5,
                    'rf_rel_theta_power': -1.3,
                    'heart_rate_87': 72.0,
                    'stillness_score': 0.7,
                    'movement_count': 3.0
                }
            },
            {
                'name': 'High Theta (Deep meditation)',
                'features': {
                    'lf_abs_theta_power': -23.0,  # High theta
                    'rf_abs_theta_power': -22.0,
                    'lf_rel_theta_power': 0.5,
                    'rf_rel_theta_power': 0.7,
                    'heart_rate_87': 58.0,
                    'stillness_score': 0.95,
                    'movement_count': 0.0
                }
            }
        ]
        
        predictions = []
        confidences = []
        
        for scenario in test_scenarios:
            result = engine.predict(scenario['features'])
            
            if result.get('ready') and result.get('prediction'):
                pred = result['prediction']
                
                predictions.append(pred['class'])
                confidences.append(pred['confidence'])
                state_label = engine.get_meditation_state_label(pred['class'])
                print(f"   {scenario['name']:30}: {state_label} (Class {pred['class']}, Conf {pred['confidence']:.3f})")
            else:
                print(f"   {scenario['name']:30}: FAILED - {result.get('error', 'Unknown error')}")
                predictions.append(-1)
                confidences.append(0.0)
        
        # Check if we get diverse predictions
        valid_predictions = [p for p in predictions if p != -1]
        unique_predictions = len(set(valid_predictions))
        avg_confidence = np.mean([c for c in confidences if c > 0]) if confidences else 0
        
        print(f"\n📊 Prediction Analysis:")
        print(f"   Valid predictions: {len(valid_predictions)}/{len(test_scenarios)}")
        print(f"   Unique classes: {unique_predictions}")
        print(f"   All predictions: {valid_predictions}")
        print(f"   Average confidence: {avg_confidence:.3f}")
        
        if len(valid_predictions) == 0:
            print(f"❌ NO PREDICTIONS WORKING - Check feature aligner errors above")
            return False
        elif unique_predictions > 1:
            print(f"🎉 SUCCESS: Model shows prediction diversity!")
            print(f"   Your normalization fix is working!")
            print(f"   Confidences look more realistic ({avg_confidence:.1%} vs 94.9%)")
            return True
        elif unique_predictions == 1:
            print(f"⚠️  Model still stuck on class {valid_predictions[0]}")
            print(f"   But at least predictions are working now!")
            print(f"   Try restarting your real-time meditation system")
            return False
        else:
            print(f"❌ Unexpected prediction pattern")
            return False
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_aligned_inference_engine()
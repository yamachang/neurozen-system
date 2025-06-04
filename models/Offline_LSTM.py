# Offline_LSTM.py 

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.inspection import permutation_importance
import tensorflow as tf
import joblib
import time

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

# Define file paths
input_file = './data/processed/cleaned_interpolated_data_nohr.csv'
output_base_dir = './models'
lstm_output_dir = os.path.join(output_base_dir, 'LSTM_Model')
feature_output_dir = os.path.join(output_base_dir, 'Feature_Importance')

# Create output directories
os.makedirs(lstm_output_dir, exist_ok=True)
os.makedirs(feature_output_dir, exist_ok=True)

# Load the cleaned and interpolated dataset
print(f"Loading data from: {input_file}")
df = pd.read_csv(input_file)
print(f"Dataset shape: {df.shape}")
print("\nMeditation state distribution:")
state_counts = df['meditation_state'].value_counts().sort_index()
print(state_counts)
state_percentages = (state_counts / len(df)) * 100
for state, count in state_counts.items():
    print(f"State {state}: {count} ({state_percentages[state]:.2f}%)")
print("\nHandling class imbalance...")

# Calculate class weights inversely proportional to class frequencies
n_samples = len(df)
class_weights = {}
for state, count in state_counts.items():
    class_weights[state] = n_samples / (len(state_counts) * count)

print("Class weights to balance the classes:")
for state, weight in class_weights.items():
    print(f"State {state}: {weight:.4f}")

# Create dictionary with integer keys for Keras
class_weight_dict = {int(k): v for k, v in class_weights.items()}

# Identify session IDs and time columns
session_col = 'session_id'
time_cols = ['epoch_start_time_original_s', 'epoch_start_time_trimmed_s']
target_col = 'meditation_state'

# List all features to use (excluding session ID, time columns, and target)
exclude_cols = [session_col] + time_cols + [target_col]
feature_cols = [col for col in df.columns if col not in exclude_cols]
print(f"\nNumber of features: {len(feature_cols)}")

# Save the feature columns for later use
with open(os.path.join(output_base_dir, 'feature_columns.txt'), 'w') as f:
    for col in feature_cols:
        f.write(f"{col}\n")

# Prepare features (X) and target (y)
X = df[feature_cols].values
y = df[target_col].values

# Scale the features
print("\nScaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler for later use
scaler_file = os.path.join(output_base_dir, 'feature_scaler.pkl')
joblib.dump(scaler, scaler_file)
print(f"Saved feature scaler to: {scaler_file}")

# Split the data into train and test sets with stratification
print("\nSplitting data into train and test sets (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Further split the training data to create a validation set
print("Further splitting training data to create validation set (60% train, 20% validation, 20% test overall)...")
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42, stratify=y_train)

print(f"Training set shape: {X_train.shape} ({X_train.shape[0] / X_scaled.shape[0]:.2%} of data)")
print(f"Validation set shape: {X_val.shape} ({X_val.shape[0] / X_scaled.shape[0]:.2%} of data)")
print(f"Test set shape: {X_test.shape} ({X_test.shape[0] / X_scaled.shape[0]:.2%} of data)")


####OFFLINE LSTM MODEL TRAINING####
print("\n" + "="*50)
print("LSTM MODEL TRAINING")
print("="*50 + "\n")

# Reshape the data for LSTM [samples, time steps, features]
X_train_reshaped = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_val_reshaped = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
X_test_reshaped = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

# Convert target to categorical format for multi-class classification
n_classes = len(np.unique(y))
y_train_cat = to_categorical(y_train, num_classes=n_classes)
y_val_cat = to_categorical(y_val, num_classes=n_classes)
y_test_cat = to_categorical(y_test, num_classes=n_classes)

print(f"Training set shape: {X_train_reshaped.shape}")
print(f"Validation set shape: {X_val_reshaped.shape}")
print(f"Test set shape: {X_test_reshaped.shape}")
print(f"Number of classes: {n_classes}")

# Build the lightweight LSTM model
print("\nBuilding lightweight LSTM model...")
model = Sequential([
    LSTM(32, input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])),
    Dropout(0.2),
    Dense(n_classes, activation='softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# Set up callbacks for early stopping and model checkpointing
model_checkpoint_file = os.path.join(output_base_dir, 'best_model.h5')
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ModelCheckpoint(model_checkpoint_file, save_best_only=True, monitor='val_accuracy')
]

# Train the model using the explicit validation set and class weights
print("\nTraining the LSTM model with class weights to address imbalance...")
start_time = time.time()
history = model.fit(
    X_train_reshaped, y_train_cat,
    epochs=100,
    batch_size=32,
    validation_data=(X_val_reshaped, y_val_cat), 
    class_weight=class_weight_dict,  
    callbacks=callbacks,
    verbose=1
)
training_time = time.time() - start_time
print(f"Training completed in {training_time:.2f} seconds")

# Save the final model
model_file = os.path.join(lstm_output_dir, 'lstm_model.h5')
model.save(model_file)
print(f"Saved model to: {model_file}")

# Load the best model for evaluation
best_model = load_model(model_checkpoint_file)

# Evaluate the model on test data
print("\nEvaluating the model on test data...")
test_loss, test_acc = best_model.evaluate(X_test_reshaped, y_test_cat, verbose=0)
print(f"Test accuracy: {test_acc:.4f}")
print(f"Test loss: {test_loss:.4f}")

# Make predictions
y_pred_prob = best_model.predict(X_test_reshaped)
y_pred = np.argmax(y_pred_prob, axis=1)

# Generate classification report
print("\nClassification Report:")
class_report = classification_report(y_test, y_pred, output_dict=True)
print(classification_report(y_test, y_pred))

# Plot training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.tight_layout()
history_plot_file = os.path.join(lstm_output_dir, 'training_history.png')
plt.savefig(history_plot_file)
print(f"Saved training history plot to: {history_plot_file}")

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
cm_file = os.path.join(lstm_output_dir, 'confusion_matrix.png')
plt.savefig(cm_file)
print(f"Saved confusion matrix to: {cm_file}")

# Calculate class-wise metrics
class_metrics = pd.DataFrame(class_report).T
class_metrics_file = os.path.join(lstm_output_dir, 'class_metrics.csv')
class_metrics.to_csv(class_metrics_file)
print(f"Saved class-wise metrics to: {class_metrics_file}")

# Generate a summary report
with open(os.path.join(lstm_output_dir, 'model_summary_report.txt'), 'w') as f:
    f.write("LSTM Model for Meditation State Classification\n")
    f.write("===========================================\n\n")
    f.write(f"Dataset shape: {df.shape}\n")
    f.write(f"Number of features: {len(feature_cols)}\n")
    f.write(f"Number of classes: {n_classes}\n\n")
    
    f.write("Model Architecture:\n")
    model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    f.write("\nTraining Information:\n")
    f.write(f"Training time: {training_time:.2f} seconds\n")
    f.write(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}\n")
    f.write(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}\n")
    f.write(f"Final training loss: {history.history['loss'][-1]:.4f}\n")
    f.write(f"Final validation loss: {history.history['val_loss'][-1]:.4f}\n\n")
    
    f.write("Test Evaluation:\n")
    f.write(f"Test accuracy: {test_acc:.4f}\n")
    f.write(f"Test loss: {test_loss:.4f}\n\n")
    
    f.write("Classification Report:\n")
    f.write(classification_report(y_test, y_pred))

print("\nLSTM model training and evaluation complete!")

####FEATURE IMPORTANCE ANALYSIS####
print("\n" + "="*50)
print("LSTM PERMUTATION FEATURE IMPORTANCE ANALYSIS")
print("="*50 + "\n")

# Permutation Importance
print("\nCalculating Permutation Importance on the LSTM model...")

# Define a scoring function 
def lstm_scorer(model, X, y):
    """Score function for the LSTM model using accuracy."""
    # Reshape X for LSTM input
    X_reshaped = X.reshape(X.shape[0], 1, X.shape[1])
    y_pred_prob = model.predict(X_reshaped, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)
    return accuracy_score(y, y_pred)

print("Starting permutation importance calculation (this may take some time)...")

perm_importance_lstm = permutation_importance(
    estimator=best_model,
    X=X_test,
    y=y_test,
    scoring=lambda estimator, X, y: lstm_scorer(estimator, X, y),
    n_repeats=10,  # Number of times to permute each feature
    random_state=42,
    n_jobs=-1 
)

# Create a DataFrame with the results
perm_importances = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': perm_importance_lstm.importances_mean,
    'Std': perm_importance_lstm.importances_std
})
perm_importances = perm_importances.sort_values('Importance', ascending=False)

# Save permutation importances to CSV
perm_imp_file = os.path.join(feature_output_dir, 'lstm_permutation_importance.csv')
perm_importances.to_csv(perm_imp_file, index=False)
print(f"Saved LSTM Permutation feature importances to: {perm_imp_file}")

# Plot permutation feature importances (top 20)
plt.figure(figsize=(12, 8))
top_20 = perm_importances.head(20)
sns.barplot(x='Importance', y='Feature', data=top_20)
plt.xlabel('Drop in Accuracy When Feature is Permuted')
plt.tight_layout()
perm_imp_plot_file = os.path.join(feature_output_dir, 'lstm_permutation_importance_plot.png')
plt.savefig(perm_imp_plot_file)
print(f"Saved LSTM Permutation importance plot to: {perm_imp_plot_file}")

# Create a more detailed visualization showing importance with error bars
plt.figure(figsize=(14, 10))
top_30 = perm_importances.head(30)
plt.errorbar(
    x=top_30['Importance'],
    y=range(len(top_30)),
    xerr=top_30['Std'],
    fmt='o',
    capsize=5
)
plt.yticks(range(len(top_30)), top_30['Feature'])
plt.xlabel('Mean Accuracy Decrease (± Std Dev)')
plt.axvline(x=0, color='red', linestyle='--', alpha=0.5)
plt.grid(True, axis='x', alpha=0.3)
plt.tight_layout()
perm_imp_detailed_file = os.path.join(feature_output_dir, 'lstm_permutation_importance_detailed.png')
plt.savefig(perm_imp_detailed_file)
print(f"Saved detailed LSTM Permutation importance plot to: {perm_imp_detailed_file}")

# Compare feature importance to feature variability
print("\nAnalyzing feature variability in relation to importance...")
# Calculate coefficient of variation for each feature
feature_stats = pd.DataFrame({
    'Feature': feature_cols,
    'Mean': X_scaled.mean(axis=0),
    'Std': X_scaled.std(axis=0)
})
feature_stats['CV'] = feature_stats['Std'] / feature_stats['Mean'].abs()
feature_stats = feature_stats.replace([np.inf, -np.inf], np.nan).fillna(0)  # Handle division by zero

# Merge with importance
feature_analysis = perm_importances.merge(feature_stats, on='Feature')
feature_analysis = feature_analysis.sort_values('Importance', ascending=False)

# Save feature analysis
feature_analysis_file = os.path.join(feature_output_dir, 'lstm_feature_analysis.csv')
feature_analysis.to_csv(feature_analysis_file, index=False)
print(f"Saved feature analysis to: {feature_analysis_file}")
print("\nLSTM Permutation Feature Importance Analysis Complete! Reports and visualizations saved.")
print("\nAll processing complete!")